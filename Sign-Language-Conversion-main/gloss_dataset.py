import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from typing import Tuple, Dict, List
from gloss_proc import GlossProcess


class GlossDataset(Dataset):
    def __init__(self):
        self.gp = GlossProcess.load_checkpoint()
        self.gdata, self.gd_labels = self.gp.get_all_gdata()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self.gp.glosses
        self.class_index = {cname: index for index,
                            cname in enumerate(self.classes)}
        self.labels = self.oh_labels(self.class_index, self.gd_labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.gdata[index]).float().to(self.device), self.labels[index]

    def oh_labels(self, class_index: Dict[str, int], gd_labels: List[str]) -> torch.Tensor:
        label_ind: torch.Tensor = torch.tensor(
            [class_index[label] for label in gd_labels])
        return one_hot(label_ind, num_classes=len(self.classes)).to(self.device)

    def get_gd_label(self, oh_label: torch.Tensor) -> str:
        return self.classes[torch.argmax(oh_label)]
