import numpy as np

def initialize_population(params):
    pop_size = params['pop_size']
    init_max_length = params['init_max_length']
    mean_length = params['mean_length']
    stddev_length = params['stddev_length']
    max_pool = params['max_pool']
    image_channel = params['image_channel']
    max_output_channel = params['max_output_channel']
    population = []
    for _ in range(pop_size):
        num_net = int(np.random.normal(mean_length, stddev_length))
        while num_net > init_max_length:
            num_net = int(np.random.normal(mean_length, stddev_length))
        num_pool = np.random.randint(0, max_pool + 1)
        num_conv = num_net - num_pool
        while num_conv <=0:
            num_pool = np.random.randint(0, max_pool + 1)
            num_conv = num_net - num_pool

        # find the position where the pooling layer can be connected
        availabel_positions = list(range(1,num_net))
        np.random.shuffle(availabel_positions)
        select_positions = np.sort(availabel_positions[0:num_pool]) # the positions of pooling layers in the net
        particle = []
        for i in range(num_net):
            if i in select_positions:
                code_pool = np.random.randint(256, 384)
                particle.append(code_pool)
            else:
                code_conv = np.random.randint(0, max_output_channel)
                particle.append(code_conv)

        population.append(particle)
    return population

def test_population():
    params = {}
    params['pop_size'] = 20
    params['init_max_length'] = 50
    params['mean_length'] = 10
    params['stddev_length'] = 1
    params['max_pool'] = 4
    params['image_channel'] = 1
    params['max_output_channel'] = 128
    pop = initialize_population(params)
    print(pop)

if __name__ == '__main__':
    test_population()
