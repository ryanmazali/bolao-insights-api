import pandas as pd

df = pd.read_csv('data/processed/fm23_player_mapping.csv')

falsos_positivos = [
    'Raúl Rangel', 'Mateo Chávez', 'Obed Vargas', 'Pavel Šulc',
    'Marcelo Flores', 'Promise David', 'Amar Memic', 'Ayoub Al-Alawi',
    'Homam Al-Amin', 'Al-Hashmi Al-Hussain', 'Sultan Al-Brake',
    'Ahmed Al-Ganehi', 'Tahsin Mohammed', 'El Kajoui', 'Salah-Eddine',
    'El Ouahdi', 'El Mourabet', 'El Aynaoui', 'Pierre Woodenski',
    'Dominique Simon', 'Lenny Joseph', 'Scott McTominay', 'Tim Weah',
    'Álex Arce', 'Jordan Bos', 'Kai Trewin', 'Lennart Karl',
    'Clément Akpa', 'Yuto Nagatomo', 'Victor Nilsson Lindelöf',
    'Sabri Ben Hassen', 'Anis Ben Slimane', 'Mohamed Hadj-Mahmoud',
    'Firas Chawat', 'Diego Moreira', 'Mohamed Hany', 'Hamza Abdel Karim',
    'Ali Nemati', 'Milad Mohammadi', 'Amir Mohammad Razagah Niya',
    'Nawaf Al Aqidi', 'Mohammed Abu Al Shamat', 'Moteb Al Harbi',
    'Alaa Al Hajji', 'Ziyad Al Johani', 'Musab Al Juwayr',
    'José María Giménez', 'El Hadji Malich Diouf', 'Ilay Camara',
    'Bara Sapoko Ndiaye', 'Ahmed Basil', 'Hussein Ali', 'Zaid Ismail',
    'Ahmed Qasim', 'Sondre Langas', 'Mohamed Amine Tougai',
    'Mohamed Amine Amoura', 'Amine Gouri', 'Abdallah Al Fakhouri',
    'Ahmad Al Juaidi', 'Mohammad Taha', 'Yousef Qashi', 'Juan Portilla',
    'Elliot Anderson', 'Joseph Anang', 'Baba Abdul Rahman',
    'Christopher Bonsu Baah', 'Brandon Thomas-Asante', 'Prince Kwabena Adu',
    'José Córdoba', 'José Fajardo', 'Lee Hanbeom', 'Kim Taehyeon',
    'Lee Taeseok', 'David Affengruber'
]

df.loc[(df['match_score'] == 85.5) & (df['supabase_name'].isin(falsos_positivos)), 'status'] = 'unmatched'
df.loc[(df['match_score'] == 85.5) & (df['supabase_name'].isin(falsos_positivos)), 'fm23_name'] = None
df.loc[(df['match_score'] == 85.5) & (df['supabase_name'].isin(falsos_positivos)), 'fm23_uid'] = None

df.to_csv('data/processed/fm23_player_mapping.csv', index=False)

matched = (df['status'] == 'matched').sum()
unmatched = (df['status'] == 'unmatched').sum()
print(f"Matched: {matched} | Unmatched: {unmatched}")