from typing import Optional

from pydantic import BaseModel


class AlgoBase(BaseModel):
    name: str = None
    num_process: int = 3
    device: str = 'cuda'
    seed: Optional[int] = 0
