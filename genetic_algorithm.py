import vectorbt as vbt
import pandas as pd
import random
import numpy as np
import empyrical as ep
import statsmodels.api as sm
from typing import List, Dict, Tuple
from signals import Signal

class GeneticAlgorithm:

    def __init__(self, hyperspace: Dict, close: pd.Series, direction: str = 'LS', 
                 asset: str = None, fitness_option: str = 'sharpe',
                 n_trades_threshold_option: str = 'off', n_trades_threshold: int = 20):
        
        self.hyperspace = hyperspace
        self.close = close
        self.direction = direction
        self.asset = asset
        self.fitness_option = fitness_option
        self.n_trades_threshold_option = n_trades_threshold_option
        self.n_trades_threshold = n_trades_threshold
        
        self.fitness_options = {
            'sharpe': 'abs(pf.returns()[algo].replace(np.inf, np.nan).replace(-np.inf, np.nan).replace(np.nan, 0).apply(ep.sharpe_ratio).median()) * (np.sign(pf.returns()[algo].replace(np.inf, np.nan).replace(-np.inf, np.nan).replace(np.nan, 0).apply(ep.cum_returns_final).median()))',
            'sharpe_min': 'pf.returns()[algo].replace(np.inf, np.nan).replace(-np.inf, np.nan).replace(np.nan, 0).apply(ep.sharpe_ratio).min()',
            'sortino': 'abs(pf.returns()[algo].replace(np.inf, np.nan).replace(-np.inf, np.nan).replace(np.nan, 0).apply(ep.sortino_ratio).median()) * (np.sign(pf.returns()[algo].replace(np.inf, np.nan).replace(-np.inf, np.nan).replace(np.nan, 0).apply(ep.cum_returns_final).median()))',
            'cagr': 'pf.returns()[algo].replace(np.inf, np.nan).replace(-np.inf, np.nan).replace(np.nan, 0).apply(ep.cagr).median()',
            'cagr_by_maxdd': 'pf.returns()[algo].replace(np.inf, np.nan).replace(-np.inf, np.nan).replace(np.nan, 0).apply(ep.cagr).div(abs(pf.returns()[algo].replace(np.inf, np.nan).replace(-np.inf, np.nan).replace(np.nan, 0).apply(ep.max_drawdown))).median()',
            'avg_trade_x_n_trades': 'trades[trades.algo == algo].Return.mean() * len(trades[trades.algo == algo])',
            'final_equity': 'pf.value()[algo].iloc[-1].median()',
            'sharpe_from_trades': "ep.sharpe_ratio(trades[trades.algo == algo].set_index('Exit Timestamp').Return.resample('1D').mean())",
            'portfolio_sharpe': "ep.sharpe_ratio(pf.returns()[algo].mean(axis=1))",
            'sharpe_v2': "ep.sharpe_ratio(trades[trades.algo == algo].groupby('Exit Timestamp').agg({'Return':'sum'}).Return)",
            'linear_fit_adj_sharpe': "ep.sharpe_ratio(pf.returns()[algo].mean(axis=1)) / (1-self._get_linear_fit(pf.value()[algo].mean(axis=1)))"
        }

    def create_random_gene(self) -> str:
        indicator = random.choice(list(self.hyperspace.keys()))
        params = {}
        for i, param in enumerate(self.hyperspace[indicator][0].keys()):
            params[f'param_{i+1}'] = random.choice(self.hyperspace[indicator][0][param])

        if isinstance(self.hyperspace[indicator][1], list):
            if len(self.hyperspace[indicator][1]) == 2:
                if isinstance(self.hyperspace[indicator][1][0], str) and self.hyperspace[indicator][1][0] == 'close':
                    func = self.hyperspace[indicator][1][0]
                    threshold = func
                else:
                    threshold = random.choice(self.hyperspace[indicator][1])
            else:
                threshold = random.choice(self.hyperspace[indicator][1])
        else:
            threshold = random.choice(self.hyperspace[indicator][1])

        if indicator in ['WEEKDAY', 'WEEK', 'MONTHDAY', 'MONTH']:
            operator = random.choice(['<', '>', '!=', '=='])
        elif indicator in ['BARPATH', 'VOL_BREAKOUT']:
            operator = '=='
        else:
            operator = random.choice(['<', '>'])

        if 'no_input' in list((self.hyperspace[indicator][0].keys())):
            return f'({indicator}() {operator} {threshold})'
        else:
            values_str = str([params[x] for x in params.keys()]).replace('[', '(').replace(']', ')')
            return f'({indicator}{values_str} {operator} {threshold})'

    def create_individual(self) -> str:
        num_entry_rules = random.randint(1, 3)
        num_exit_rules = random.randint(1, 3)

        entry_rules = ' & '.join([self.create_random_gene() for _ in range(num_entry_rules)])
        exit_rules = ' & '.join([self.create_random_gene() for _ in range(num_exit_rules)])

        return entry_rules + ' $ ' + exit_rules

    def create_symmetrical_individual(self) -> str:
        num_rules = random.randint(1, 3)
        entry_rules = []
        exit_rules = []

        for _ in range(num_rules):
            entry_rule = self.create_random_gene()
            entry_rules.append(entry_rule)
            exit_rule = entry_rule.replace('<', '>') if '<' in entry_rule else entry_rule.replace('>', '<')
            exit_rules.append(exit_rule)

        return ' & '.join(entry_rules) + ' $ ' + ' & '.join(exit_rules)

    def create_limited_individual(self, population: List[str] = [], excluded_gene: str = '@@@') -> str:
        while True:
            num_entry_genes = 1
            num_exit_genes = 1
            
            try:
                entry_rules = [self.create_random_gene() for _ in range(num_entry_genes)]
                exit_rules = [self.create_random_gene() for _ in range(num_exit_genes)]
            
                individual = ' & '.join(entry_rules) + ' $ ' + ' & '.join(exit_rules)

                if (individual not in population) and (excluded_gene not in individual):
                    return individual
            except AttributeError:
                raise AttributeError("self.hyperspace must be a dictionary, not a tuple")

    def tournament_selection(self, population: List[str], fitness_scores: List[float], 
                           tournament_size: int, num_to_select: int) -> List[str]:
        selected = []
        for _ in range(num_to_select):
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def crossover_multigene(self, individual1: str, individual2: str) -> str:
        entry1, exit1 = individual1.split(' $ ')
        entry2, exit2 = individual2.split(' $ ')

        entry_genes1 = entry1.split(' & ')
        entry_genes2 = entry2.split(' & ')
        exit_genes1 = exit1.split(' & ')
        exit_genes2 = exit2.split(' & ')

        entry_crossover_point = random.randint(1, min(len(entry_genes1), len(entry_genes2)))
        exit_crossover_point = random.randint(1, min(len(exit_genes1), len(exit_genes2)))

        new_entry_genes = entry_genes1[:entry_crossover_point] + entry_genes2[entry_crossover_point:]
        new_exit_genes = exit_genes1[:exit_crossover_point] + exit_genes2[exit_crossover_point:]

        return ' & '.join(new_entry_genes) + ' $ ' + ' & '.join(new_exit_genes)

    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        parent1_rules = parent1.split(' $ ')
        parent2_rules = parent2.split(' $ ')

        crossover_point = random.randint(1, min(len(parent1_rules), len(parent2_rules)) - 1)

        offspring1 = ' $ '.join(parent1_rules[:crossover_point] + parent2_rules[crossover_point:])
        offspring2 = ' $ '.join(parent2_rules[:crossover_point] + parent1_rules[crossover_point:])

        return offspring1, offspring2

    def crossover_symmetrical(self, individual1: str, individual2: str) -> str:
        entry1, _ = individual1.split(' $ ')
        entry2, _ = individual2.split(' $ ')

        entry_genes1 = entry1.split(' & ')
        entry_genes2 = entry2.split(' & ')

        entry_crossover_point = random.randint(1, min(len(entry_genes1), len(entry_genes2)))
        new_entry_genes = entry_genes1[:entry_crossover_point] + entry_genes2[entry_crossover_point:]

        new_exit_genes = []
        for gene in new_entry_genes:
            exit_gene = gene.replace('<', '>') if '<' in gene else gene.replace('>', '<')
            new_exit_genes.append(exit_gene)

        return ' & '.join(new_entry_genes) + ' $ ' + ' & '.join(new_exit_genes)

    def mutate(self, individual: str, mutation_rate: float) -> str:
        if random.random() < mutation_rate:
            entry, exit = individual.split(' $ ')

            if random.random() < 0.5:
                entry_genes = entry.split(' & ')
                gene_to_mutate = random.randint(0, len(entry_genes) - 1)
                entry_genes[gene_to_mutate] = self.create_random_gene()
                entry = ' & '.join(entry_genes)
            else:
                exit_genes = exit.split(' & ')
                gene_to_mutate = random.randint(0, len(exit_genes) - 1)
                exit_genes[gene_to_mutate] = self.create_random_gene()
                exit = ' & '.join(exit_genes)

            individual = entry + ' $ ' + exit

        return individual

    def mutate_symmetrical_individual(self, individual: str, mutation_rate: float) -> str:
        entry_signal, exit_signal = individual.split(' $ ')
        entry_rules = entry_signal.split(' & ')
        exit_rules = exit_signal.split(' & ')

        for i in range(len(entry_rules)):
            if random.random() < mutation_rate:
                mutated_entry_rule = self.create_random_gene()
                entry_rules[i] = mutated_entry_rule
                mutated_exit_rule = mutated_entry_rule.replace('<', '>') if '<' in mutated_entry_rule else mutated_entry_rule.replace('>', '<')
                exit_rules[i] = mutated_exit_rule

        return ' & '.join(entry_rules) + ' $ ' + ' & '.join(exit_rules)

    def fitness_vectorized(self, population: List[str]) -> List[float]:
        entries = {}
        exits = {}
        _price = {}
        for individual in population:
            entries[individual] = eval(individual.split('$')[0])
            exits[individual] = eval(individual.split('$')[1])
            _price[individual] = self.close
            
        entries = pd.concat(entries, axis=1, join='outer').fillna(False)
        exits = pd.concat(exits, axis=1, join='outer').fillna(False)
        _price = pd.concat(_price, axis=1, join='outer')
            
        if self.direction == 'L':
            pf = vbt.Portfolio.from_signals(_price, entries=entries, exits=exits, fees=0.001)
        elif self.direction == 'S':
            pf = vbt.Portfolio.from_signals(_price, short_entries=entries, short_exits=exits, fees=0.001)
        elif self.direction == 'LS':
            pf = vbt.Portfolio.from_signals(_price, entries=entries, short_entries=exits, fees=0.001)
        
        return [pf.returns()[algo].apply(ep.sharpe_ratio).replace(np.nan, 0).median() for algo in population]

    def fitness_parametrized(self, population: List[str]) -> List[float]:
        entries = {}
        exits = {}
        _price = {}
        for individual in population:
            i = self.explode_individual(individual)
            entries[individual] = eval(i.split('$')[0])
            exits[individual] = eval(i.split('$')[1])
            _price[individual] = self.close.copy()
        entries = pd.concat(entries, axis=1, join='outer').fillna(False)
        exits = pd.concat(exits, axis=1, join='outer').fillna(False)
        _price = pd.concat(_price, axis=1, join='outer')
        
        entries.index.name = 'date'
        exits.index.name = 'date'
        _price.index.name = 'date'
        
        if self.direction == 'L':
            pf = vbt.Portfolio.from_signals(_price, entries=entries, exits=exits, fees=0.001)
        elif self.direction == 'S':
            pf = vbt.Portfolio.from_signals(_price, short_entries=entries, short_exits=exits, fees=0.001)
        elif self.direction == 'LS':
            pf = vbt.Portfolio.from_signals(_price, entries=entries, short_entries=exits, fees=0.001)
        
        trades = pf.trades.records_readable
        trades = trades[trades.Status == 'Closed']
        trades['algo'] = [x[0] for x in trades.Column]

        fitness_scores = []
        for algo in population:
            
            if self.n_trades_threshold_option == 'on':
                if len(trades[trades.algo == algo]) > self.n_trades_threshold:
                    fit = eval(self.fitness_options[self.fitness_option])
                else:
                    fit = np.nan
            else:   
                fit = eval(self.fitness_options[self.fitness_option])
            
            fitness_scores.append(fit)
            
        return fitness_scores

    def _get_linear_fit(self, equity_series: pd.Series) -> float:
        X = np.array(range(len(equity_series)))
        X = sm.add_constant(X)
        y = equity_series.values
        model = sm.OLS(y, X)
        results = model.fit()
        return results.rsquared

    def explode_individual(self, individual: str) -> str:
        """
        Convert individual string to evaluatable expression with Signals class.
        
        Args:
            individual: String containing entry/exit rules separated by '$'
            
        Returns:
            Modified string that can be evaluated to get entry/exit Series
        """
        # Split into entry and exit parts
        entry_part, exit_part = individual.split('$')
        entry_part = entry_part.strip()
        exit_part = exit_part.strip()
        
        # Add Signals class wrapper to each function call
        for part in [entry_part, exit_part]:
            # Handle & separated rules
            rules = part.split('&')
            for i, rule in enumerate(rules):
                rule = rule.strip()
                rules[i] = rule.replace('(', '(Signal(close).', 1) #!TODO: Adjust if hyperspace is not close_only
                rules[i] = rules[i].replace('close', 'self.close')
            
            if part == entry_part:
                entry_part = ' & '.join(rules)
            else:
                exit_part = ' & '.join(rules)

        # Recombine with '$' separator
        return f'{entry_part} $ {exit_part}'

