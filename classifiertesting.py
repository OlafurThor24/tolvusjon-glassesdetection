import torch
from glasses_detector import GlassesClassifier, GlassesDetector

# classifier = GlassesClassifier()
# classifier.process_dir(
# input_path="1.jpg",
# format={True: "1", False: "0"},
# show=True,
# )

classifier = GlassesClassifier(kind="eyeglasses")

classifier.process_dir(
    input_path="glasses-noglasses/validate/glasses",         # failed files will raise a warning
    output_path="glasses-noglasses/validate/glasses/output.csv", # img_name1.jpg,<pred>...
    format="proba",                   # <pred> is a probability of sunglasses
    pbar="Processing",                # set to None to disable
)


# detector = GlassesDetector()
# with torch.inference_mode():
#     detector.process_file(
#         input_path="1.jpg",
#         format="img",
#         show=True,
#     )
