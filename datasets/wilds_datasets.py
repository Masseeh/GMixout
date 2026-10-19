import torch
import os
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.fmow_dataset import FMoWDataset


class WILDSDataset(torch.utils.data.Dataset):
    classnames_txt = ""

    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

        self.classnames = self.get_classnames()
        self.num_classes = self.dataset.n_classes
        self.cls_num_list = self.get_cls_num_list()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y, _ = self.dataset[index]        
        return x, y, index

    def get_classnames(self):
        classnames = []

        if isinstance(self.classnames_txt, str) and os.path.isfile(self.classnames_txt):
            with open(self.classnames_txt, 'r') as f:
                for line in f.readlines():
                    classnames.append(line.strip())
        elif isinstance(self.classnames_txt, list):
            classnames = [c for c in self.classnames_txt]
        else:
            raise ValueError("classnames_txt must be a path to a txt file or a list of class names.")

        return classnames
    
    def get_cls_num_list(self):
        counter = [0] * self.num_classes
        for label in self.dataset.y_array:
            counter[label] += 1
        return counter

class WILDSiWildCam(WILDSDataset):
    classnames_txt = "./datasets/iWildCam/categories.txt"

    def __init__(self, dataset):
        super().__init__(dataset)

    def eval(self, y_pred, y_true):
        return self.dataset.eval(y_pred, y_true, None)

class WILDSFMoW(WILDSDataset):
    classnames_txt = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
                    "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site",
                    "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble",
                    "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station",
                    "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station",
                    "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse",
                    "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility",
                    "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge",
                    "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack",
                    "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening",
                    "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]

    def __init__(self, dataset):
        super().__init__(dataset)

    def eval(self, y_pred, y_true):
        return self.dataset.eval(y_pred, y_true, self.dataset.dataset.metadata)

class WILDSiWildCam_ID(WILDSiWildCam):
    def __init__(self, root, split="train", transform=None, cfg=None):
        dataset = IWildCamDataset(root_dir=root)
        if "val" in split:
            split = "id_val"
        elif "test" in split:
            split = "id_test"
        super().__init__(dataset.get_subset(split, transform=transform))
    
class WILDSiWildCam_OOD(WILDSiWildCam):
    def __init__(self, root, split="train", transform=None, cfg=None):
        dataset = IWildCamDataset(root_dir=root)
        if "val" in split:
            split = "id_val"
        super().__init__(dataset.get_subset(split, transform=transform))

class WILDSiWildCam_Oracle(WILDSiWildCam):
    def __init__(self, root, split="train", transform=None, cfg=None):
        dataset = IWildCamDataset(root_dir=root)
        super().__init__(dataset.get_subset(split, transform=transform))

class WILDSFMoW_ID(WILDSFMoW):
    def __init__(self, root, split="train", transform=None, cfg=None):
        dataset = FMoWDataset(root_dir=root)
        if "val" in split:
            split = "id_val"
        elif "test" in split:
            split = "id_test"
        super().__init__(dataset.get_subset(split, transform=transform))

class WILDSFMoW_OOD(WILDSFMoW):
    def __init__(self, root, split="train", transform=None, cfg=None):
        dataset = FMoWDataset(root_dir=root)
        if "val" in split:
            split = "id_val"
        super().__init__(dataset.get_subset(split, transform=transform))