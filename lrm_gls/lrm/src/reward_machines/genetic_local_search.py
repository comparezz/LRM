import random, time
from deap import base, creator, tools, algorithms
from reward_machines.ts_util import evaluate_neighborhood,evaluate_rm, rm2str
from reward_machines.gls_util import evaluate_neighborhood_gls,evaluate_rm, rm2str

def sample_random_rm(U_max, observations, initial_obs):
    delta  = {}
    for i in range(U_max):
        if i == 0:
            available = [o for o in observations if o not in initial_obs]
        else:
            available = [o for o in observations if len([1 for k in range(i) if (k,o) in delta and delta[(k,o)] == i]) == 0]

        for j in range(i+1, U_max):
            if len(available) > 0:
                o = random.choice(available)
                delta[(i,o)] = j
                available.remove(o)
    return delta

def create_individual():
    return creator.Individual(sample_random_rm(U_max, observations, initial_obs))

def run_genetic_local_search(traces, U_max, population_size, n_workers, lr_steps, current_rm, perfect_rm):

        start = time.time()
        steps = 0

        #  Precomputing auxiliary variables for learning the reward machine
        initial_obs = set([trace[0][0] for trace in traces])
        observations = list(set([o for trace in traces for o,_ in trace])) # the order is important for the tabu list
        N = dict([(o,set()) for o in observations])
        for trace in traces:
            for t in range(1,len(trace)):
                N[trace[t-1][0]].add(trace[t][0])
        
        # NOTE: Evaluating the current RM (we change the RM only if we find a better RM)
        current_rm_cost = float('inf')
        if current_rm is not None:
            current_rm_cost,_,_ = evaluate_rm(current_rm, None, set(), U_max, observations, N, traces)

        # NOTE: Evaluating the perfect RM (just to know)
        perfect_rm_cost,_,_ = evaluate_rm(perfect_rm, None, set(), U_max, observations, N, traces)

        print("\nParameters")
        print("U_max:", U_max)
        print("Num obs:", len(observations))
        print("Num traces:", len(traces))
        print("Num workers:", n_workers)
        print("Neighborhood:", len(observations)*U_max*(U_max-1)/2)
        print("initial_obs:", initial_obs)
        print()

        # Learning an RM for this traces using tabu search (and starting from a random RM)
        best_cost  = float('inf') 
        best_delta = None
        population_set   = set()
        gls_queue = [] # use append and pop(0)!
        while steps <= lr_steps:
            # 2. Creating an initial random RM
            delta = sample_random_rm(U_max, observations, initial_obs)
                    
            # 3. Evaluate current RM
            cost, delta, delta_str = evaluate_rm(delta, None, population, U_max, observations, N, traces)
            if delta is None:
                # The generated RM is tabu
                continue

            if cost < best_cost:
                best_cost  = cost
                best_delta = delta        

        # 4. Add the RM to the tabu list
        # update_tabu_list(delta_str, tabu_set, tabu_queue, tabu_list_max)

        while steps <= lr_steps:
            print("%0.2f[m]\t%d\tPerfect RM: %0.2f\tOld RM: %0.2f\tBest: %0.2f\tCurrent: %0.2f"%((time.time() - start)/60, steps, perfect_rm_cost, current_rm_cost, best_cost, cost))
            cost, delta, delta_str = evaluate_neighborhood(n_workers, delta, population_set, U_max, observations, N, initial_obs, traces)
                # Create fitness (more is better)
            creator.create('FitnessMax', base.Fitness, weights=(1.0,))
            # Create a container to hold the individuals 
            creator.create('Individual', list, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()
            toolbox.register('individual', create_individual)
            toolbox.register('population', tools.initRepeat, list, toolbox.individual)

            # Register an evaluate function
            toolbox.register('evaluate', evaluate_rm)
            # Register a cross-over strategy
            toolbox.register('mate', tools.cxTwoPoint)
            # Register a mutation strategy
            toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.05)
            # Register a selection strategy
            toolbox.register('select', tools.selTournament, tournsize=3)

            population = toolbox.population(n=200)
            fits = toolbox.map(toolbox.evaluate, population)
            for fit, ind in zip(fits, population):
                ind.fitness.values = fit


            ngen=100
            for g in range(ngen):
                population = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
                fits = toolbox.map(toolbox.evaluate, population)
                for fit, ind in zip(fits, population):
                    ind.fitness.values = fit

            population = toolbox.select(population, k=len(population))

            if delta is None:
                    print("No way out! restarting the search")
                    break

            # updating the best delta if needed
            if cost < best_cost:
                best_cost  = cost
                best_delta = delta  
                
            steps += 1

            # Testing the reward machine (no input label is used also as output)
            for i in range(U_max):
                for o in observations:
                    if (i,o) in delta:
                        j = delta[(i,o)]
                        assert (j,o) not in delta, "The RM has a bug!"

            # Testing the reward machine (no initial transition given the initial observations)
            for o in initial_obs:
                if (0,o) in delta:
                    assert False, "The RM has a transition in u=0 using the first observation of the trace!"

            # Testing the reward machine (no useless transitions)
            to_remove = set([(i,o) for i,o in delta])
            for trace in traces:
                u1 = 0
                for t in range(1, len(trace)):
                    o2,_ = trace[t]
                    if (u1,o2) in delta:
                        to_remove.discard((u1,o2))
                        u1 = delta[(u1,o2)]
            assert len(to_remove) == 0, "The RM has useless transitions!"
        # returning None if the best RM is not better than our current RM
        if current_rm_cost <= best_cost:
            return None, current_rm_cost, perfect_rm_cost
        # Otherwise, we return the best found RM
        return best_delta, best_cost, perfect_rm_cost
    