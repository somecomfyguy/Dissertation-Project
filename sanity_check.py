from modules.dataset_module.process_texbat import scan_texbat_segments
segs = scan_texbat_segments("./modules/dataset_module/datasets/TexbatSpoofing")
from collections import Counter
print(Counter(s.label for s in segs))