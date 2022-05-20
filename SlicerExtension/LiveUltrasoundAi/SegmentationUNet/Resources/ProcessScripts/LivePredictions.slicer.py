import pickle
import sys
import traceback

import numpy as np
import PyTorchUtils
import slicer
from PIL import Image

torch = PyTorchUtils.PyTorchUtilsLogic().torch  # will be installed if necessary
if torch.cuda.is_available():
    device = torch.device("cuda", 0)  # automatically, single card TODO
else:
    device = torch.device("cpu")

ACTIVE = bytes([1])
NOT_ACTIVE = bytes([0])

# While debigging this script, you may add a convenient full path to the log file name to find it easily.
# Otherwise, find the log file in your Slicer binaries installation folder.
# Here is an example how to print debug info in the log file:
# f.write("resizeInputArray, input_array shape: {}\n".format(input_array.shape))

f = open(r"LivePredictionsLog.txt", "a")


def resizeInputArray(input_array):
    """
  Resize the image and add it to the current buffer
  :param input_image: np.array
  :return: np.array resized and rescaled for AI model
  """
    input_image = Image.fromarray(
        input_array
    )  # PIL.Image knows np.array reverses axes: M -> image.width, F -> image.height
    resized_input_array = np.array(
        input_image.resize(
            (
                int(
                    input_image.width * slicer_to_model_scaling[1]
                ),  # M direction (width on US machine)
                int(
                    input_image.height * slicer_to_model_scaling[0]
                ),  # F direction (height on US machine)
            ),
            resample=Image.BILINEAR,
        )
    )

    """
    resized_input_array = np.flip(
        resized_input_array, axis=0
    )  # We trained on images with opposite sound direction
    resized_input_array = (
        resized_input_array / resized_input_array.max()
    )  # Scaling intensity to 0-1
    """
    resized_input_array = np.expand_dims(
        resized_input_array, axis=0
    )  # Add Channel dimension
    resized_input_array = np.expand_dims(
        resized_input_array, axis=0
    )  # Add Batch dimension
    return torch.tensor(resized_input_array, dtype=torch.float32, device=device)


def resizeOutputArray(y):
    output_array = y[0, 0, :, :].cpu().numpy()  # (F, M) TODO
    """
    output_array = np.flip(
        output_array, axis=0
    )  # This flip should match the one in the other resize function
    """
    apply_logarithmic_transformation = True
    logarithmic_transformation_decimals = 4
    if apply_logarithmic_transformation:
        e = logarithmic_transformation_decimals
        output_array = np.log10(np.clip(output_array, 10 ** (-e), 1.0) * (10 ** e)) / e
    output_image = Image.fromarray(output_array)  # image.height -> F, image.width -> M
    upscaled_output_array = np.array(
        output_image.resize(
            (
                int(output_image.width * model_to_slicer_scaling[1]),
                int(output_image.height * model_to_slicer_scaling[0]),
            ),
            resample=Image.BILINEAR,
        )
    )
    upscaled_output_array = upscaled_output_array * 255  # (F, M)
    upscaled_output_array = np.clip(upscaled_output_array, 0, 255)

    return upscaled_output_array


# Do first read at the same time that we instantiate the model, this ensures that the graph is written/ready right away when we start live predictions.
try:
    input = sys.stdin.buffer.read(1)  # Read control byte
    if input == ACTIVE:
        input_length = sys.stdin.buffer.read(8)  # Read data length
        input_length = int.from_bytes(input_length, byteorder="big")
        input_data = sys.stdin.buffer.read(input_length)  # Read the data
    elif input == NOT_ACTIVE:
        f.close()
        sys.exit()

    input_data = pickle.loads(
        input_data
    )  # Axes: (F, M) because slicer.util.array reverses axis order from IJK to KJI

    # Load AI model TODO
    modelPath = slicer.modules.segmentationunet.path.replace(
        "SegmentationUNet.py", "model"
    )
    sys.path.append(modelPath)  # May not need
    try:
        from unet import UNet

        model = UNet()  # TODO
    except:
        f.write(
            "Could not import model from file: {}\n".format(modelPath + "/unet.py")
        )

    try:
        model.load_state_dict(
            torch.load(input_data["model_path"], map_location="cpu")
        )
        model.to(device)
        model.eval()

    # Determine resize factors
    input0_shape = model.get_inputs()  # (b,c,h,w) TODO
    model_input_shape = (input0_shape[1], input0_shape[2], input0_shape[3])

    # f.write("model_input_shape:  {}\n".format(str(model_input_shape)))
    # f.write("input array (volume) shape: {}\n".format(str(input_data['volume'].shape)))

    slicer_to_model_scaling = (
        model_input_shape[1] / input_data["volume"].shape[0],  # F image direction
        model_input_shape[2] / input_data["volume"].shape[1],  # M image direction
    )
    model_to_slicer_scaling = (
        input_data["volume"].shape[0] / model_input_shape[1],
        input_data["volume"].shape[1] / model_input_shape[2],
    )

    # Run a dummy prediction to initialize
    input_array = resizeInputArray(input_data["volume"])
    _ = model(input_array)

    # Starting loop to receive images until status set to NOT_ACTIVE
    while True:
        input = sys.stdin.buffer.read(1)  # Read control byte
        # f.write("Active status: {}\n".format(str(input)))

        if input == ACTIVE:
            input_length = sys.stdin.buffer.read(8)  # Read data length
            input_length = int.from_bytes(input_length, byteorder="big")
            input_data = sys.stdin.buffer.read(input_length)  # Read the data
        else:
            break

        input_data = pickle.loads(input_data)
        input_array = input_data["volume"]
        input_transform = input_data["transform"]
        if input_array.shape[0] == 1:
            input_array = input_array[0, :, :]
        input_array = resizeInputArray(input_array)

        y = model(input_array)
        # f.write("Prediction shape in while loop: {}\n".format(y.shape))
        output_array = resizeOutputArray(y)

        output = {}
        output["prediction"] = output_array
        output["transform"] = np.array(
            input_transform
        )  # Make a copy of the input transform, because it has already lapsed in Slicer
        pickled_output = pickle.dumps(output)

        sys.stdout.buffer.write(pickled_output)
        sys.stdout.buffer.flush()

    f.write("Exiting by request from Slicer\n")
    f.close()
    sys.exit()

except:
    f.write("ERROR in process script:\n{}\n".format(traceback.format_exc()))
    f.close()
    sys.exit()
