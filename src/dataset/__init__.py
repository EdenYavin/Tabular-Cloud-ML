from src.dataset.heloc import HelocDataset
from src.dataset.adult import AdultDataset
from src.dataset.gesture_phase import GesturePhaseDataset
from src.dataset.bank_marketing import BankMarketing
from src.dataset.students_dropout import StudentsDropout
from src.dataset.rain import RainDataset
from src.dataset.load_approval import LoadApprovalDataset
from src.dataset.airline import AirlineSatisfaction
from src.dataset.base import RawDataset


DATASETS = {
    HelocDataset.name: HelocDataset,
    GesturePhaseDataset.name: GesturePhaseDataset,
    AdultDataset.name: AdultDataset,
    BankMarketing.name: BankMarketing,
    StudentsDropout.name: StudentsDropout,
    RainDataset.name: RainDataset,
    LoadApprovalDataset.name: LoadApprovalDataset,
    AirlineSatisfaction.name: AirlineSatisfaction,
}