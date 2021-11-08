import numpy as np

def fpso(cur_particle, gbest, pbest, velocity, params):

    cur_len = len(cur_particle)
    pbest_len = len(pbest)
    gbest_len = len(gbest)

    # 1.particle alignment
    offset1 = np.random.randint(0, abs(cur_len - pbest_len) + 1)
    if pbest_len >= cur_len:
        new_pbest = np.asarray(pbest[offset1:offset1 + cur_len])
    else:
        new_pbest = np.zeros(cur_len)
        new_pbest[offset1:offset1 + pbest_len] = pbest

    offset2 = np.random.randint(0, abs(cur_len - gbest_len) + 1)
    if gbest_len >= cur_len:
        new_gbest = np.asarray(gbest[offset2:offset2 + cur_len])
    else:
        new_gbest = np.zeros(cur_len)
        new_gbest[offset2:offset2 + gbest_len] = gbest

    # 2.velocity calculation
    w, c1, c2 = 0.7298, 1.49618, 1.49618
    r1 = np.random.random(cur_len)
    r2 = np.random.random(cur_len)
    new_velocity = np.asarray(velocity) * w + c1 * r1 * (new_pbest - cur_particle) + c2 * r2 * (new_gbest - cur_particle)

    # 3.particle updating
    new_particle = list(map(int, cur_particle + new_velocity))  #particle里面的数必须为整数
    new_velocity = list(new_velocity)

    # 4.architecture evolving
    while len(new_particle) > 1 and (new_particle[0] < 0 or new_particle[0] > 255):
        del new_particle[0]
        del new_velocity[0]

    if new_particle[0] < 0 or new_particle[0] > 255:
        new_particle[0] = 1

    j = 0
    while j < len(new_particle):
        if new_particle[j] < 0:
            del new_particle[j]
            del new_velocity[j]
            j -= 1
        elif 128 <= new_particle[j] <= 255:
            value1 = new_particle[j]
            new_particle[j] = value1 // 2
            new_particle.insert(j + 1, value1 - value1 // 2)

            value2 = new_velocity[j]
            new_velocity[j] = value2 // 2
            new_velocity.insert(j + 1, value2 - value2 // 2)
        elif 384 <= new_particle[j] <= 511:
            # if dimen falls into this range, we will have two different pooling types, avg+max
            value1 = new_particle[j]
            gap = value1 - 383
            new_particle[j] = 319 + gap - gap//2   # avg pooling
            new_particle.insert(j + 1, 320 - gap//2)    # max pooling

            value2 = new_velocity[j]
            new_velocity[j] = value2 - value2 // 2
            new_velocity.insert(j + 1, value2 // 2)
        while new_particle[j] > 511:
            value1 = new_particle[j]
            new_particle[j] = value1 // 2
            new_particle.insert(j + 1, value1 - value1 // 2)    # both are with the same pooling type

            value2 = new_velocity[j]
            new_velocity[j] = value2 // 2
            new_velocity.insert(j + 1, value2 - value2 // 2)
            j -= 1
        j += 1

    pool_num = _calculate_pool_numbers(new_particle)
    while pool_num > params['max_pool']:
        new_particle, new_velocity = cut_pool(new_particle, pool_num - params['max_pool'], new_velocity)
        pool_num = _calculate_pool_numbers(new_particle)

    return new_particle, new_velocity

def _calculate_pool_numbers(particle):
    num_pool = 0
    for dimension in particle:
        if 256<=dimension<=383:
            num_pool+=1
    return num_pool

def cut_pool(particle, num2cut, velocity):
    pool_idx = [list(enumerate(particle))[i][0] for i in range(len(particle)) if 256 <= particle[i] <= 383]
    selected_idx = np.random.choice(pool_idx, num2cut, replace=False)
    particle = [particle[i] for i in range(len(particle)) if i not in selected_idx]
    velocity = [velocity[i] for i in range(len(velocity)) if i not in selected_idx]
    return particle, velocity
