import numpy as np
import random
import copy
import argparse

def distribution(prob, repetition):
    probs = prob.split(" ")
    l = len(probs)
    probs = [float(p) for p in probs]
    probs_interval = np.zeros(l)

    for i in range(l):
        if i == 0:
            probs_interval[i] = probs[i]
        else:
            probs_interval[i] = probs_interval[i-1]+ probs[i]

    get_item = np.zeros(repetition)
    for i in range(repetition):
        r = random.random()
        for j in range(l):
            if r <= probs_interval[j]:
                get_item[i] = j
                break

    get_item = [int(item) for item in get_item]
    get_result = [str(item) for item in get_item]
    get_result = "\n".join(get_result)

    return get_result

def strategy(strategy, state, crowded, repetitions):
    strategys = strategy.split(" ")
    strategys = [float(stg) for stg in strategys]
    l = len(strategys)
    h = int(strategys[0])
    p = []
    a = []
    b = []
    for i in range(1, l):
        if i % (1 + h * 2) == 1:
            p.append(strategys[i])
        elif 1 < i % (1 + h * 2) < 2 + h:
            a.append(strategys[i])
        else:
            b.append(strategys[i])
    # states
    if crowded == 1:
        a = a[int((state + 1) * len(p) - len(p)): int((state + 1) * len(p))]
        a = [str(prob) for prob in a]
        a = ' '.join(a)
        state = distribution(a, repetitions)
    else:
        b = b[int((state + 1) * len(p) - len(p)): int((state + 1) * len(p))]
        b = [str(prob) for prob in b]
        b = ' '.join(b)
        state = distribution(b, repetitions)

    state = state.split("\n")
    states = [int(s) for s in state]
    # decision
    decision = []
    for s in states:
        r = random.random()
        if r < p[s]:
            decision.append(1)
        else:
            decision.append(0)

    d_s = list(zip(decision,states))

    return d_s


def exercise2(strategy):
    for i in range(len(strategy)):
        print(str(strategy[i][0]) + "\t" + str(strategy[i][1]))


def initial(pop_size, h):
    pop = []
    for i in range(pop_size):
        individual = []
        individual.append(h)
        for i in range(h):
            individual.append(random.random())
            for i in range(h):
                a_value = np.random.dirichlet(np.ones(h), size=1)
                a = (a_value.tolist())[0]
            individual.extend(copy.copy(a))
            for i in range(h):
                b_value = np.random.dirichlet(np.ones(h), size=1)
                b = (b_value.tolist())[0]
            individual.extend(copy.copy(b))

        individuals = [str(x) for x in individual]
        individuals = ' '.join(individuals)
        pop.append(individuals)
        individual.clear()

    return pop


def fitness(result):
    fit = []
    l = len(result)
    attendance = result.count(1)
    
    if l == 0:
        crowded = 0
    else:
        crowded = attendance / l

    if crowded < 0.6:
        for i in range(l):
            if result[i] == 1:
                fit.append(1)
            else:
                fit.append(0)

        flag = 0
    else:
        for i in range(l):
            if result[i] == 0:
                fit.append(1)
            else:
                fit.append(0)
        flag = 1

    return fit, flag, attendance


def getM(h, parent_1, parent_2):
    parent_1 = parent_1.split(" ")
    parent_1 = [float(p) for p in parent_1]
    l1 = len(parent_1)
    parent_2 = parent_2.split(" ")
    parent_2 = [float(p) for p in parent_2]
    l2 = len(parent_2)

    p1 = []
    p2 = []
    a1 = []
    a2 = []
    b1 = []
    b2 = []

    for i in range(1, l1):
        if i % (1 + h * 2) == 1:
            p1.append(parent_1[i])
        elif 1 < i % (1 + h * 2) < 2 + h:
            a1.append(parent_1[i])
        else:
            b1.append(parent_1[i])
    for i in range(1, l2):
        if i % (1 + h * 2) == 1:
            p2.append(parent_2[i])
        elif 1 < i % (1 + h * 2) < 2 + h:
            a2.append(parent_2[i])
        else:
            b2.append(parent_2[i])

    return p1, p2, a1, a2, b1, b2


def mutation(h, p, a, b):
    prob = random.random()
    if prob > 0.99:
        p.clear()
        for i in range(h):
            p.append(random.random())
    index_a = int(random.random() * h)

    prob_a = random.random()
    if prob_a > 0.99:
        a_value = np.random.dirichlet(np.ones(h), size=1)
        component_a = (a_value.tolist())[0]
        for i in range(h):
            a[(index_a * h) + i] = component_a[i]

    index_b = int(random.random() * h)
    prob_b = random.random()
    if prob_b > 0.99:
        b_value = np.random.dirichlet(np.ones(h), size=1)
        component_b = (b_value.tolist())[0]
        for i in range(h):
            b[(index_b * h) + i] = component_b[i]

    return p, a, b


def crossover(pop_size, h, p1, p2, a1, a2, b1, b2):
    new_pop = []
    # crossover and mutation
    for i in range(pop_size):
        for j in range(len(p1)):
            prob = random.random()
            if prob > 0.45:
                p1[j], p2[j] = p2[j], p1[j]
        for j in range(0, len(a1), h):
            prob = random.random()
            if prob > 0.45:
                a1[j:j + h], a2[j:j + h] = a2[j:j + h], a1[j:j + h]
        for j in range(0, len(b1), h):
            prob = random.random()
            if prob > 0.45:
                b1[j:j + h], b2[j:j + h] = b2[j:j + h], b1[j:j + h]

        new1 = []
        new1.append(h)
        new2 = []
        new2.append(h)

        for i in range(h):
            new1.append(p1[i])
            new2.append(p2[i])
            new1.extend(copy.copy(a1[i * h:(i * h) + h]))
            new2.extend(copy.copy(a2[i * h:(i * h) + h]))
            new1.extend(copy.copy(b1[i * h:(i * h) + h]))
            new2.extend(copy.copy(b2[i * h:(i * h) + h]))

        prob = random.random()
        if prob > 0.45:
            new1 = [str(x) for x in new1]
            new1 = ' '.join(new1)

            new_pop.append(new1)
        else:
            new2 = [str(x) for x in new2]
            new2 = ' '.join(new2)
            new_pop.append(new2)

    return new_pop


def Coevolution(pop_size, h, weeks, max_t):
    pop = initial(pop_size, h)
    state = np.zeros(pop_size).tolist()
    result = np.zeros(pop_size).tolist()
    fit, flag, attendance = fitness(result)

    all_result = []

    for iteration in range(max_t):
        fit = np.zeros(pop_size).tolist()
        for w in range(weeks):
            d_s = []
            for i in range(len(pop)):
                next_ds = strategy(pop[i], state[i], flag, 1)
                d_s.append(next_ds)
            d = []
            s = []
            for ds in d_s:
                temp_d = ds[0][0]
                temp_s = ds[0][1]
                d.append(temp_d)
                s.append(temp_s)
            state.clear()
            result.clear()
            attendance_situation = copy.copy(d)
            state = copy.copy(s)
            d.clear()
            s.clear()

            new_fit, flag, attendance = fitness(attendance_situation)

            for i in range(len(fit)):
                fit[i] = fit[i] + (copy.copy(new_fit[i]))
            new_fit.clear()

            result = []
            result.append(w)
            result.append(iteration)
            result.append(attendance)
            result.append(flag)
            result.extend(copy.copy(attendance_situation))
            all_result.append(copy.copy(result))

        m = 0
        s = 0
        count = 0

        for i in range(len(fit)):
            if fit[i] > fit[m]:
                m = i
            else:
                if fit[i] > fit[s]:
                    s = i
            count = count + fit[i]

        parent_1 = copy.copy(pop[m])
        parent_2 = copy.copy(pop[s])

        p1, p2, a1, a2, b1, b2 = getM(h, parent_1, parent_2)

        p1, a1, b1 = mutation(h, p1, a1, b1)
        p2, a2, b2 = mutation(h, p2, a2, b2)

        pop.clear()
        pop = crossover(pop_size, h, p1, p2, a1, a2, b1, b2)

    return all_result


def exercise3(all_result):
    for result in all_result:
        result = [str(x) for x in result]
        result = '\t'.join(result)
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='start', add_help=False)
    parser.add_argument('-hpush', type=int, default=10)
    parser.add_argument('-question', type=int, default=3)
    parser.add_argument('-lambdapush', type=int, default=700)
    parser.add_argument('-max_t', type=int, default=20)
    parser.add_argument('-repetitions', type=int, default=5)
    parser.add_argument('-weeks', type=int, default=10)
    parser.add_argument('-prob', type=str, default='0 0 1 0')
    parser.add_argument('-strategy', type=str, default='2 0.1 0.0 1.0 1.0 0.0 1.0 0.9 0.1 0.9 0.1')
    parser.add_argument('-state', type=int, default=1)
    parser.add_argument('-crowded', type=int, default=0)
    parser.add_argument('-crossrat', type=float, default=0)
    parser.add_argument('-mutatrat', type=float, default=0)
    

    args = parser.parse_args()
    question = args.question

    if question == 1:
        #distribution (prob, repetition)
        result_1 = distribution(args.prob, args.repetitions)
        print(result_1)

    elif question == 2:
        #exercise2(strategy,state,crowded,repetitions)
        #strategy(strategy, state, crowded, repetitions)
        result_2 = strategy(args.strategy, args.state, args.crowded, args.repetitions)
        exercise2(result_2)

    else:
        # Coevolution(pop_size, h, weeks, max_t)
        result_3 = Coevolution(args.lambdapush, args.hpush, args.weeks, args.max_t)
        exercise3(result_3)
