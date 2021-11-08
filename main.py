from utils import Utils, Log
from population import initialize_population
from evaluate import decode, fitnessEvaluate
from evolve import fpso
import copy, os

def create_directory():
    dirs = ['./log', './populations', './scripts', './datasets']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def fitness_evaluate(population, curr_gen):
    filenames = []
    for i, particle in enumerate(population):
        filename = decode(particle, curr_gen, i)
        filenames.append(filename)

    acc_set = fitnessEvaluate(filenames, curr_gen, is_test=False)
    return acc_set

def evolve(population, gbest_individual, pbest_individuals, velocity_set, params):
    offspring = []
    new_velocity_set = []
    for i,particle in enumerate(population):
        new_particle, new_velocity = fpso(particle, gbest_individual, pbest_individuals[i], velocity_set[i], params)
        offspring.append(new_particle)
        new_velocity_set.append(new_velocity)
    return offspring, new_velocity_set

def update_best_particle(population, acc_set, gbest, pbest):
    if not pbest:
        pbest_individuals = copy.deepcopy(population)
        pbest_accSet = copy.deepcopy(acc_set)
        gbest_individual, gbest_acc = getGbest([pbest_individuals, pbest_accSet])
    else:
        gbest_individual, gbest_acc = gbest
        pbest_individuals, pbest_accSet = pbest
        for i,acc in enumerate(acc_set):
            if acc > pbest_accSet[i]:
                pbest_individuals[i] = copy.deepcopy(population[i])
                pbest_accSet[i] = copy.deepcopy(acc)
            if acc > gbest_acc:
                gbest_individual = copy.deepcopy(population[i])
                gbest_acc = copy.deepcopy(acc)

    return [gbest_individual, gbest_acc], [pbest_individuals, pbest_accSet]

def getGbest(pbest):
    pbest_individuals, pbest_accSet = pbest
    gbest_acc = 0
    gbest = None
    for i,indi in enumerate(pbest_individuals):
        if pbest_accSet[i] > gbest_acc:
            gbest = copy.deepcopy(indi)
            gbest_acc = copy.deepcopy(pbest_accSet[i])
    return gbest, gbest_acc

def fitness_test(gbest_individual):
    filename = decode(gbest_individual, -1, -1)
    acc_set = fitnessEvaluate([filename], -1, is_test=True)
    return acc_set[0]

def evolveCNN(params):
    gen_no = 0
    Log.info('Initialize...')
    population = initialize_population(params)
    # Utils.save_population('pop', population, gen_no)

    Log.info('EVOLVE[%d-gen]-Begin evaluate the fitness' % (gen_no))
    acc_set = fitness_evaluate(population, gen_no)
    Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))

    # gbest
    [gbest_individual, gbest_acc], [pbest_individuals, pbest_accSet] = update_best_particle(population, acc_set, gbest=None, pbest=None)
    Log.info('EVOLVE[%d-gen]-Finish the updating' % (gen_no))

    Utils.save_population_and_acc('population', population, acc_set, gen_no)
    Utils.save_population_and_acc('pbest', pbest_individuals, pbest_accSet, gen_no)
    Utils.save_population_and_acc('gbest', [gbest_individual], [gbest_acc], gen_no)

    gen_no += 1
    velocity_set = []
    for ii in range(len(population)):
        velocity_set.append([0]*len(population[ii]))

    for curr_gen in range(gen_no, params['num_iteration']):
        params['gen_no'] = curr_gen

        Log.info('EVOLVE[%d-gen]-Begin to pso evolution' % (curr_gen))
        population, velocity_set = evolve(population, gbest_individual, pbest_individuals, velocity_set, params)
        Log.info('EVOLVE[%d-gen]-Finish pso evolution' % (curr_gen))

        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (curr_gen))
        acc_set = fitness_evaluate(population, curr_gen)
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (curr_gen))

        [gbest_individual, gbest_acc], [pbest_individuals, pbest_accSet] = update_best_particle(population, acc_set, gbest=[gbest_individual, gbest_acc], pbest=[pbest_individuals, pbest_accSet])
        Log.info('EVOLVE[%d-gen]-Finish the updating' % (curr_gen))

        Utils.save_population_and_acc('population', population, acc_set, curr_gen)
        Utils.save_population_and_acc('pbest', pbest_individuals, pbest_accSet, curr_gen)
        Utils.save_population_and_acc('gbest', [gbest_individual], [gbest_acc], curr_gen)

    # final training and test on testset
    gbest_acc = fitness_test(gbest_individual)
    num_parameters = Utils.calc_parameters_num(gbest_individual)
    Log.info('The acc of the best searched CNN architecture is [%.5f], number of parameters is [%d]' % (gbest_acc, num_parameters))
    Utils.save_population_and_acc('final_gbest', [gbest_individual], [gbest_acc], -1)

if __name__ == '__main__':
    create_directory()
    params = Utils.get_init_params()
    evoCNN = evolveCNN(params)

