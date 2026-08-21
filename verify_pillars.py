from cadgenesis import get_pillar_overview
from cadgenesis.evaluation.benchmark_runner import run_pillar_benchmark

print(len(get_pillar_overview()))
print(run_pillar_benchmark().to_dict())
