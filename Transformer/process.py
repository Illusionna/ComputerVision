import os
from utils.Seed.seed import SEED
from utils.Preprocess.Process_Interface import PROCESS

os.system('cls')

seedValue = 42
seed = SEED(seedValue)
seed.SeedEverything()

pro = PROCESS(
    seed = seedValue,
    datasetsPath = './datasets'
)

pro.Split()
pro.Save()