from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np

def get_precentiles(a):
    p25 = float(np.percentile(a, 25))
    p50 = float(np.percentile(a, 50))
    p75 = float(np.percentile(a, 75))
    return p25, p50, p75

def export_results():
    num_lines = 0
    num_files = 0
    agents = ["lrm-dqn", "lrm-qrm", "dqn", "human"]
    worlds = ["cookieworld", "keysworld", "symbolworld"]

    for agent in agents:
        for world in worlds:
            # Searching for the results computed this far for (agent,world)
            if agent.startswith("lrm-"): folder = "../results/LRM/%s/%s/"%(agent.replace("lrm-",""), world)
            else:                        folder = "../results/BASELINE/%s/%s/"%(agent, world)
            
            if not exists(folder):
                continue
            
            files = [join(folder, f) for f in listdir(folder) if f.endswith("_reward.txt") and isfile(join(folder, f))]
            if len(files) == 0:
                continue
            
            # Reading all the results computed so far
            results = {}
            for f_path in files:
                f = open(f_path)
                for l in f:
                    if len(l.strip()) > 0:
                        step,reward = tuple(l.strip().split("\t"))
                        step,reward = int(step), float(reward)
                        if step not in results:
                            results[step] = []
                        results[step].append(reward)
                f.close()
            
            # Saving a summary for the results found this far
            f_out = "../results/summary/%s/"%world
            if not exists(f_out): makedirs(f_out)
            f_out += "%s.txt"%agent
            f = open(f_out,'w')
            steps = sorted(results.keys())
            for step in steps:
                p25, p50, p75 = get_precentiles(results[step])
                s = [str(step), "%0.2f"%p25, "%0.2f"%p50, "%0.2f"%p75]
                f.write("\t".join(s) + "\n")
            f.close()

if __name__ == '__main__':
    # EXAMPLE: python3 export_results
    export_results()