import time
import pandas as pd
import random
import math
import numpy as np
from scipy.spatial import distance_matrix
import os

# Чтение файлов
def file_reader(file_path):
    data = {
        'name': '',
        'track_no': 0,
        'optimal_val': 0,
        'type': '',
        'dimension': 0,
        'capacity': 0,
        'node_coords': {},
        'demands': {},
        'depot': None,
    }
    
    section = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('NAME'):
                data['name'] = line.split(':')[1].strip()
            elif line.startswith('COMMENT'):
                comment = line.strip()

                track_no = ''
                for i in range(40, len(comment)):
                    symb = comment[i]
                    if symb == ',':
                        break
                    track_no = track_no + symb
                data['track_no'] = int(track_no)

                optimal_val = ''
                for i in range(2, len(comment)):
                    symb = comment[len(comment)-i]
                    if symb == ' ':
                        break
                    optimal_val = symb + optimal_val
                data['optimal_val'] = int(optimal_val)
                
            elif line.startswith('TYPE'):
                data['type'] = line.split(':')[1].strip()
            elif line.startswith('DIMENSION'):
                data['dimension'] = int(line.split(':')[1].strip())
            elif line.startswith('CAPACITY'):
                data['capacity'] = int(line.split(':')[1].strip())
            elif line == 'NODE_COORD_SECTION':
                section = 'NODE_COORD'
                continue
            elif line == 'DEMAND_SECTION':
                section = 'DEMAND'
                continue
            elif line == 'DEPOT_SECTION':
                section = 'DEPOT'
                continue
            elif line == 'EOF':
                break
                
            if section == 'NODE_COORD':
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = int(parts[1])
                    y = int(parts[2])
                    data['node_coords'][node_id] = (x, y)
                    
            elif section == 'DEMAND':
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    data['demands'][node_id] = demand
                    
            elif section == 'DEPOT':
                if line.strip() == '-1':
                    continue
                depot_id = int(line.strip())
                data['depot'] = depot_id
                
    return data

# Случайные решения
def random_solution(n, m):

    m -= 1 # Так мы будем уверены, что путей не больше, чем машин
    
    total_length = n + m - 1  # Общая длина списка
    
    result = [None] * total_length
    
    remaining_ones = m
    
    # Размещаем единицы так, чтобы они не стояли в соседних ячейках
    available_positions = [i for i in range(1, total_length) if i != 1]
    while remaining_ones > 0 and available_positions:
        pos = random.choice(available_positions)
        result[pos] = 1
        remaining_ones -= 1
        
        neighbors = {pos - 1, pos, pos + 1}
        available_positions = [p for p in available_positions if p not in neighbors]
    
    # Заполняем оставшиеся позиции числами от 2 до n
    numbers = list(range(2, n + 1))
    random.shuffle(numbers)
    
    num_index = 0
    for i in range(total_length):
        if result[i] is None:
            result[i] = numbers[num_index]
            num_index += 1
    
    return result

# Целевая функция
def calc_result(used_solution, capacity, dist_matrix, demands, base_penalty):

    solution = used_solution.copy()

    solution.append(1)
    solution.insert(0, 1)

    z = 0
    demand = 0
    for i in range(1, len(solution)):

        z += dist_matrix[solution[i]-1][solution[i-1]-1]

        if solution[i] == 1:
            if demand > capacity:
                penalty = (demand-capacity)**2 / capacity * base_penalty
            else:
                penalty = 0

            z += penalty
            demand = 0
        else:
            demand += demands[solution[i]] 

    return z

# Селекция
def selection(solutions, beta):
    fitness = np.exp(-beta * np.array(solutions['z_s']))
    total = np.sum(fitness)
    
    if total == 0:
        probabilities = np.ones(len(solutions['paths'])) / len(solutions['paths'])
    else:
        probabilities = fitness / total
    
    probabilities /= np.sum(probabilities)
    selected_idx = np.random.choice(len(solutions['paths']), p=probabilities)
    return selected_idx

# Мутация
def mutation(prev_solution):
    solution = prev_solution.copy()
    n = len(solution)
    l = max(1, round(n * 0.1))
    
    # Два непересекающихся подсписка
    i = random.randint(0, n - l)
    j = random.randint(0, n - l)
    
    while abs(i - j) < l:
        j = random.randint(0, n - l)
    
    sublist1 = solution[i:i+l]
    sublist2 = solution[j:j+l]
    
    # С вероятностью 30% отзеркаливаем один из подсписков
    if random.random() < 0.3:
        if random.choice([True, False]):
            sublist1 = sublist1[::-1]
        else:
            sublist2 = sublist2[::-1]
    
    solution[i:i+l] = sublist2
    solution[j:j+l] = sublist1
    
    return solution

# Функция для записи решения в файл
def write_solution_to_file(folder, file_name, best_result):

    output_file = os.path.join(f"{folder}_results", file_name.replace('.vrp', '.sol'))
    
    routes = []
    current_route = []
    
    for node in best_result['solution']:
        if node == 1:
            routes.append(current_route)
            current_route = []
        else:
            current_route.append(node - 1)
    
    if current_route:
        routes.append(current_route)
    
    with open(output_file, 'w') as f:
        for i, route in enumerate(routes, 1):
            route_str = ' '.join(map(str, route))
            f.write(f"Route #{i}: {route_str}\n")
        
        f.write(f"cost {best_result['z']}\n")

# Сам алгоритм
def bee_colony_algorithm(data, solutions_num, base_penalty, sol_stognation, best_stognation, beta):
    # Матрица расстояний и общий путь
    coords = np.array([data['node_coords'][node_id] for node_id in sorted(data['node_coords'])])
    dist_matrix = distance_matrix(coords, coords)
    total_path = sum(sum(row) for row in dist_matrix) // 2

    solutions = {
        'paths': [],
        'z_s': [],
        'stognations': []
    }
    
    # Первые решения
    for i in range(solutions_num):
        solutions['paths'].append(random_solution(data['dimension'], data['track_no']))
        z = calc_result(solutions['paths'][i], data['capacity'], dist_matrix, data['demands'], base_penalty)
        solutions['z_s'].append(z)
        solutions['stognations'].append(0)

    min_z = min(solutions['z_s'])
    best_result = {
        'solution': solutions['paths'][solutions['z_s'].index(min_z)],
        'z': round(min_z),
        'stognation': 0
    }

    print('Итерации:', end=' ')

    # Поиск лучшего решения
    for iteration in range(best_stognation):
        selected_id = selection(solutions, beta)
        new_sol = mutation(solutions['paths'][selected_id])
        new_z = calc_result(new_sol, data['capacity'], dist_matrix, data['demands'], base_penalty)

        if new_z < solutions['z_s'][selected_id]:
            solutions['paths'][selected_id] = new_sol
            solutions['z_s'][selected_id] = new_z
        else:
            solutions['stognations'][selected_id] += 1

            if solutions['stognations'][selected_id] == sol_stognation:
                solutions['paths'][selected_id] = random_solution(data['dimension'], data['track_no'])
                solutions['z_s'][selected_id] = calc_result(
                    solutions['paths'][selected_id], data['capacity'], dist_matrix, data['demands'], base_penalty)
                solutions['stognations'][selected_id] = 0

        min_z = min(solutions['z_s'])
        if min_z < best_result['z']:
            best_result = {
                'solution': solutions['paths'][solutions['z_s'].index(min_z)],
                'z': round(min_z),
                'stognation': 0
            }
            # Динамическое обновление строки
            print(f'\rЛучшее значение: {best_result["z"]} (итерация {iteration + 1}/{best_stognation})', end='', flush=True)
        else:
            best_result['stognation'] = iteration

    print()  # Переход на новую строку после завершения
    return best_result

# Обработка файлов
def process_files(file_list, folder, results_file, solutions_num, base_penalty, sol_stognation, best_stognation, beta):

    df = pd.read_csv(results_file)
    
    for file_name in file_list:
        print(f'\n===== Обработка {file_name} ================')
        data = file_reader(f'{folder}\\{file_name}')

        start  = time.time()
        
        best_result = bee_colony_algorithm(data, solutions_num, base_penalty, sol_stognation, best_stognation, beta)

        # Запись данных в таблицу
        work_time = round(time.time() - start, 6)
        
        idx = df.index[df['name'] == data['name']].tolist()
        
        if best_result['z'] < (df.loc[idx[0], 'best_value']):
            df.loc[idx[0], 'best_value'] = best_result['z']
            df.loc[idx[0], 'time'] = work_time

            # Созраняем файл в папку
            write_solution_to_file(folder, file_name, best_result)

        print(f'\nОбработка завершена')
        print(f'Время: {work_time}')
    
    df.to_csv(f'{results_file}', index=False)

    return 0

# Названия файлов
A = [
    "A-n32-k5.vrp", "A-n33-k5.vrp", "A-n33-k6.vrp",
    "A-n34-k5.vrp", "A-n36-k5.vrp", "A-n37-k5.vrp",
    "A-n37-k6.vrp", "A-n38-k5.vrp", "A-n39-k5.vrp",
    "A-n39-k6.vrp", "A-n44-k6.vrp", "A-n45-k6.vrp",
    "A-n45-k7.vrp", "A-n46-k7.vrp", "A-n48-k7.vrp",
    "A-n53-k7.vrp", "A-n54-k7.vrp", "A-n55-k9.vrp",
    "A-n60-k9.vrp", "A-n61-k9.vrp", "A-n62-k8.vrp",
    "A-n63-k9.vrp", "A-n63-k10.vrp", "A-n64-k9.vrp",
    "A-n65-k9.vrp", "A-n69-k9.vrp", "A-n80-k10.vrp"
]

B = [
    "B-n31-k5.vrp", "B-n34-k5.vrp", "B-n35-k5.vrp",
    "B-n38-k6.vrp", "B-n39-k5.vrp", "B-n41-k6.vrp",
    "B-n43-k6.vrp", "B-n44-k7.vrp", "B-n45-k5.vrp",
    "B-n45-k6.vrp", "B-n50-k7.vrp", "B-n50-k8.vrp",
    "B-n51-k7.vrp", "B-n52-k7.vrp", "B-n56-k7.vrp",
    "B-n57-k7.vrp", "B-n57-k9.vrp", "B-n63-k10.vrp",
    "B-n64-k9.vrp", "B-n66-k9.vrp", "B-n67-k10.vrp",
    "B-n68-k9.vrp", "B-n78-k10.vrp"
]

param_grid = {
    'solutions_num': [100, 150, 175, 200, 250, 300],
    'base_penalty': [100, 200, 300, 400, 500],
    'sol_stognation': [1000, 1500, 2000, 2500, 3000, 3500],
    'best_stognation': [100000, 200000, 300000, 400000,500000],
    'beta': [0.1, 0.5, 1.0]
}

import optuna

def objective(trial):
    solutions_num = trial.suggest_int('solutions_num', 50, 400)
    base_penalty = trial.suggest_int('base_penalty', 100, 400)
    sol_stognation = trial.suggest_int('sol_stognation', 1000, 4000)
    best_stognation = trial.suggest_int('best_stognation', 300000, 500000)
    beta = trial.suggest_float('beta', 0.1, 1.0)

    # Загрузка тестовых данных (например, одного файла)
    data = file_reader('A/A-n32-k5.vrp')
    best_result = bee_colony_algorithm(data, solutions_num, base_penalty, sol_stognation, best_stognation, beta)
    return best_result['z']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print(study.best_params)