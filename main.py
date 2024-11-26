from genetic_algorithm import GeneticAlgorithm
from hyperspaces import close_only_hyperspace as hyperspace
from search import run_genetic_search


if __name__ == '__main__':
    poplogs = run_genetic_search(asset='BTCUSDT',
                                n_trades_threshold=20,
                                tournament_size=3,
                                direction='L',
                                hyperspace=hyperspace,
                                population_size=100,
                                num_generations=10000,  
                                close_data_path='./data/binance_futures_close_1d.csv',
                                data_interval='1d',
                                individual_func='limited')