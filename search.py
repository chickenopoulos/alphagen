from genetic_algorithm import GeneticAlgorithm
from hyperspaces import close_only_hyperspace as hyperspace
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

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
    individual_func='limited', # 'limited' or 'unlimited'
    save_progression_gif=False
):
    # Create poplogs filename from parameters
    # poplogs_filename = f"{asset}_{direction}_{fitness_option}_{population_size}_{data_interval}_{individual_func}.csv"
    poplogs_filename = 'test_poplogs.csv'

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

        # Update population logs
        poplogs = pd.concat([poplogs, fit])


    if save_progression_gif:

        print('Generating progression gif...')

        # Create frames directory
        os.makedirs("frames", exist_ok=True)

        # Initialize plot
        fig, ax = plt.subplots(figsize=(15, 7))
        lines = []
        frames = []

        best_individual_overall = poplogs.nlargest(1, 'fitness_score').iloc[0]
        best_fitness_so_far = -10e10
        best_individual_so_far = None

        # Create frames for each generation
        for generation in range(num_generations-1):

            # Get best individual for current generation
            top = poplogs[poplogs.generation == generation].nlargest(1, 'fitness_score').iloc[0]
            current_individual = top.individual
            current_fitness = top.fitness_score
            
            # Plot current performance
            pf = ga.get_individual_performance(current_individual)
            equity_curve = pf.value()[asset]
            
            # Plot with default gray color and low alpha
            line, = ax.plot(equity_curve, lw=2, alpha=0.3, color='gray')
            lines.append(line)
            
            # Set all lines to gray with low alpha
            for line in lines[:-1]:
                line.set_color('gray')
                line.set_alpha(0.1)
            
            if current_fitness >= best_fitness_so_far:
                best_fitness_so_far = current_fitness
                best_individual_so_far = current_individual
                
                # Make best performer green with full opacity
                lines[-1].set_color('black')
                lines[-1].set_alpha(1.0)
                lines[-1].set_zorder(len(lines))
            else:
                # Plot best individual so far in green
                best_line = ax.plot(equity_curve, lw=2, alpha=1.0, color='black')[0]
                lines.append(best_line)
                
                # Set all other lines to gray with low alpha
                for line in lines[:-1]:
                    line.set_color('gray')
                    line.set_alpha(0.1)
            
            plt.title(f'Generation: {generation} | Best so far: {best_individual_so_far}')
            
            frame_filename = f"frames/frame_{generation}.png"
            plt.savefig(frame_filename)
            frames.append(frame_filename)

        # Add final frame showing best overall individual
        pf = ga.get_individual_performance(best_individual_overall.individual)
        equity_curve = pf.value()[asset]

        # Set all existing lines to gray with low alpha
        for line in lines:
            line.set_color('gray')
            line.set_alpha(0.1)

        # Plot best overall individual in green
        best_line = ax.plot(equity_curve, lw=2, alpha=1.0, color='green')[0]
        plt.title(f'Best Overall: {best_individual_overall.individual}')

        frame_filename = f"frames/frame_{num_generations}.png"
        plt.savefig(frame_filename)
        frames.append(frame_filename)
            
        # Create GIF
        gif_filename = "./progression.gif"
        with imageio.get_writer(gif_filename, mode="I", duration=1) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                writer.append_data(image)
                
        # Cleanup frames
        for frame in frames:
            os.remove(frame)
        plt.close()

    poplogs.to_csv(f'{poplogs_filename}')

    return ga, poplogs
