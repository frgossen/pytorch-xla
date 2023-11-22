import json 
import os 
import re 
import collections
import numpy as np 
from scipy.stats.mstats import gmean


path = "./output-w-natural-batch-sizes"

def dict_maker():
  return collections.defaultdict(dict_maker)
 


def load_results_jsonl():
  with open(path + "/results.jsonl", mode="r", encoding="utf-8") as f:
    return [json.loads(line) for line in f.read().splitlines()]



def load_dumps():

  data = dict_maker()
  for dname in os.listdir(path):
    m = re.match(r"(?P<name>[a-zA-Z0-9_-]+)-(?P<xla>None|PJRT)-(?P<dynamo>inductor|openxla|openxla_eval)-(?P<test>eval|train)-profile", dname)
    if m is None: 
      continue
    name = m.group("name")
    xla = m.group("xla")
    dynamo = m.group("dynamo")
    test = m.group("test")
    assert (xla == "None" and dynamo == "inductor") or (xla == "PJRT" and dynamo == "openxla" and test == "train") or (xla == "PJRT" and dynamo == "openxla_eval" and test == "eval")

    dpath = path + "/" + dname
    ddata = dict_maker()
    for fname in os.listdir(dpath):

      fpath = dpath + "/" + fname

      if fname.startswith("pt-profile"):
        m = re.match(r"pt-profile-(?P<repeat>[0-9]+).txt", fname)
        assert m is not None
        repeat = int(m.group("repeat"))
        with open(fpath, mode="r", encoding="utf-8") as f:
          ddata[repeat]["pt-profile"] = f.read()

      if fname.startswith("ptxla-metrics"):
        m = re.match(r"ptxla-metrics-(?P<repeat>[0-9]+).txt", fname)
        assert m is not None
        repeat = int(m.group("repeat"))
        with open(fpath, mode="r", encoding="utf-8") as f:
          ddata[repeat]["ptxla-metrics"] = f.read()

    data[name][xla][dynamo][test] = ddata

  return data









def load_dumps_with_results_jsonl():

  data = load_dumps()

  results_jsonl = load_results_jsonl()
  for jsonline in results_jsonl:
    name = jsonline["model"]["model_name"]
    xla = jsonline["experiment"]["xla"]
    dynamo = jsonline["experiment"]["dynamo"]
    test = jsonline["experiment"]["test"]

    data[name][xla][dynamo][test]["batch_size"] = jsonline["experiment"]["batch_size"]

    data[name][xla][dynamo][test]["error"] = None
    data[name][xla][dynamo][test]["median_total_time"] = None
    data[name][xla][dynamo][test]["median_per_iter_time"] = None
    data[name][xla][dynamo][test]["median_trace_per_iter_time"] = None

    if "error" in jsonline["metrics"]:
      data[name][xla][dynamo][test]["error"] = jsonline["metrics"]["error"]
    else:
      data[name][xla][dynamo][test]["median_total_time"] = np.median(jsonline["metrics"]["total_time"])
      data[name][xla][dynamo][test]["median_per_iter_time"] = np.median(jsonline["metrics"]["per_iter_time"])
      if xla:
        data[name][xla][dynamo][test]["median_trace_per_iter_time"] = np.median(jsonline["metrics"]["trace_per_iter_time"])

  return data




def load_ptxla_inductor_comparison():

  raw_data = load_dumps_with_results_jsonl()
  data = dict_maker()
  
  for name in raw_data:
    xla_train = raw_data[name]["PJRT"]["openxla"]["train"]
    xla_eval = raw_data[name]["PJRT"]["openxla_eval"]["eval"]
    inductor_train = raw_data[name][None]["inductor"]["train"]
    inductor_eval = raw_data[name][None]["inductor"]["eval"]

    data[name]["train"]["xla"] = xla_train
    data[name]["eval"]["xla"] = xla_eval
    data[name]["train"]["inductor"] = inductor_train
    data[name]["eval"]["inductor"] = inductor_eval

    for test in ["train", "eval"]:
      data[name][test]["median_speedup"] = None
      inductor_median_total_time = data[name][test]["inductor"]["median_total_time"]
      xla_median_total_time = data[name][test]["xla"]["median_total_time"]

      if inductor_median_total_time is not None and xla_median_total_time is not None:
        data[name][test]["median_speedup"] = inductor_median_total_time / xla_median_total_time

      data[name][test]["batch_size"] = [data[name][test]["inductor"]["batch_size"], data[name][test]["xla"]["batch_size"]]
      if len(set(data[name][test]["batch_size"])) == 1:
        data[name][test]["batch_size"] = data[name][test]["batch_size"][0]


  return data



def filter_by_speedup(data, fn):
  filtered = dict_maker()
  for name in data:
    for test in data[name]:
      if fn(data[name][test]["median_speedup"]):
        filtered[name][test] = data[name][test] 
  return filtered

      


def geomean_speedup(data, test):
  speedups = []
  for name in data:
    s = data[name][test]["median_speedup"] 
    if s is not None:
      speedups.append(s)
  return gmean(speedups)


def fully_supported_names(data, test):
  names = []
  for n in data:
    if data[n][test]["median_speedup"] is not None:
      names.append(n)
  return names








# results_jsonl = load_results_jsonl()
# print(results_jsonl[0])


# dumps = load_dumps()



# dumps = load_dumps_with_results_jsonl()



# def filter_dumps(name, xla, dynamo, test, repeat, key):
#   filtered = make_dict()
#   for n in dumps:
#     if name is not None and name != n:
#       continue
#     for x in dumps[n]:
#       for d in dumps[n][x]:
#         for t in dumps[n][x][d]:
#           for r in dumps[n][x][d][t]:
          




# print(dumps["BERT_pytorch"]["PJRT"]["openxla_eval"]["eval"]["median_total_time"])



data = load_ptxla_inductor_comparison()


print(data["cm3leon_generate"]["eval"]["median_speedup"])

# print(geomean_speedup(data, "train"))
# print(fully_supported_names(data, "test"))
