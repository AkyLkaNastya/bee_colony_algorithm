import time
import pandas as pd

# Чтение файлов
def file_reader(file_path):
    data = {
        'name': '',
        'comment': '',
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
                data['comment'] = line.split(':')[1].strip()
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

def process_files(file_list, folder, results_file):

    df = pd.read_csv(results_file)
    
    for file_name in file_list:
        cvrp_data = file_reader(f'{folder}\\{file_name}')

        print(f'===== Обработка {file_name} ================')
        start = time.time()


        ############################################################

        work_time = round(time.time() - start, 6)
        
        idx = df.index[df['name'] == cvrp_data['name']].tolist()
        
        df.loc[idx[0], 'time'] = work_time
        df.loc[idx[0], 'value'] = '-'

        print(f'Обработка завершена\n')
        print(f'Время: {work_time}')
        print(f'---------------------------------------------\n\n')
    
    df.to_csv(f'{folder}_results\{results_file}', index=False)

# Обработка папки "А"
process_files(A, 'A', 'A_results.csv')

# Обработка папки "B"
process_files(B, 'B', 'B_results.csv')