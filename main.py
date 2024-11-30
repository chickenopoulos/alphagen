from genetic_algorithm import GeneticAlgorithm
import hyperspaces
from search import run_genetic_search


if __name__ == '__main__':
    poplogs = run_genetic_search(
                asset='EURUSD=X',
                oos_timestamp='2023-01-01',
                mutation_rate=0.2,
                fitness_option='sharpe',
                n_trades_threshold_option='on',
                n_trades_threshold=100,
                tournament_size=3,
                direction='L',
                hyperspace=hyperspaces.close_only_hyperspace,
                population_size=1000,
                num_generations=10000,
                close_data_path='./data/EURUSD_YFINANCE_1D.csv',
                data_interval='1d',
                individual_func='unlimited' # 'limited' or 'unlimited'
            )