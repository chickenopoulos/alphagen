from genetic_algorithm import GeneticAlgorithm
from hyperspaces import close_only_hyperspace as hyperspace
import pandas as pd
import random
from tqdm import tqdm

def run_genetic_search(
    asset='BTCUSDT',
    oos_timestamp='2023-01-01',
    mutation_rate=0.2,
    fitness_option='linear_fit_adj_sharpe',
    n_trades_threshold_option='on',
    n_trades_threshold=10,
    tournament_size=3,
    direction='L',
    hyperspace=hyperspace,
    population_size=100,
    num_generations=100,
    close_data_path='../binance_futures_close_1d.csv',
    data_interval='1d',
    individual_func='limited' # 'limited' or 'unlimited'
):
    # Create poplogs filename from parameters
    poplogs_filename = f"{asset}_{direction}_{fitness_option}_{population_size}_{data_interval}_{individual_func}.csv"
    
    # Load OHLCV data
    df = pd.read_csv(close_data_path)
    df.set_index('time', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.loc[:oos_timestamp]
    df.dropna(inplace=True)
    close = df.Close.to_frame(name=asset)
    _open = df.Open.to_frame(name=asset)
    high = df.High.to_frame(name=asset)
    low = df.Low.to_frame(name=asset)
    volume = df.Volume.to_frame(name=asset)

    # Initialize GA
    ga = GeneticAlgorithm(
        hyperspace=hyperspace,
        close=close,
        _open=_open,
        high=high,
        low=low,
        volume=volume,
        direction=direction,
        asset=asset,
        fitness_option=fitness_option,
        n_trades_threshold_option=n_trades_threshold_option,
        n_trades_threshold=n_trades_threshold
    )

    # Initialize population
    if individual_func == 'limited':
        generate_individual = ga.create_limited_individual
    else:
        generate_individual = ga.create_individual
    population = [generate_individual() for _ in range(population_size)]
    fitness_scores = ga.fitness_parametrized(population)
    fit = pd.DataFrame({
        'individual': population,
        'fitness_score': fitness_scores
    })

    print('Initial population fitness scores:')
    print(fit.fitness_score.describe())

    # Run GA
    poplogs = pd.DataFrame()
    for generation in tqdm(range(num_generations)):

        # Selection
        selected = ga.tournament_selection(population, fitness_scores, tournament_size, population_size // 2)

        # Crossover
        offspring = []
        while len(offspring) < population_size - len(selected):
            parent1, parent2 = random.sample(selected, 2)
            children = ga.crossover(parent1, parent2)
            for child in children:
                if child not in population:
                    offspring.append(child)
        population = selected + offspring

        # Mutation
        population = [ga.mutate(ind, mutation_rate) for ind in population]

        # Replace duplicates with new unique random individuals
        clean_population = pd.Series(population).drop_duplicates(keep='first').values.tolist()
        num_duplicates = population_size - len(clean_population)
        population = clean_population + [generate_individual() for _ in range(num_duplicates)]

        # Evaluate new generation
        fitness_scores = ga.fitness_parametrized(population)
        fit = pd.DataFrame({
            'individual': population,
            'fitness_score': fitness_scores,
            'generation': [generation] * len(population)
        })

        print(f'Generation {generation} fitness scores:')
        print(fit.fitness_score.describe())

        # Update population logs
        poplogs = pd.concat([poplogs, fit])

    poplogs.to_csv(f'./results/{poplogs_filename}')
    return poplogs

