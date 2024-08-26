from src.dataset.raw.base import RawDataset
from src.dataset.raw.heloc import HelocDataset
from src.dataset.raw.adult import AdultDataset
from src.dataset.raw.gesture_phase import GesturePhaseDataset
from src.dataset.raw.bank_marketing import BankMarketing

DATASETS = {
    HelocDataset.name: HelocDataset,
    GesturePhaseDataset.name: GesturePhaseDataset,
    AdultDataset.name: AdultDataset,
    BankMarketing.name: BankMarketing
}