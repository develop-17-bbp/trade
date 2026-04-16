"""
Genetic Strategy Evolution Loop
Runs on a schedule to evolve the population of DNA strategies.
Designed for RTX 5090 with large population sizes.

Usage:
    python -m src.scripts.genetic_loop --population_size 100 --interval 2
"""
import argparse
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [GENETIC] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def run_evolution_cycle(population_size: int):
    """Run one full genetic evolution cycle for BTC and ETH."""
    try:
        from src.trading.genetic_strategy_engine import GeneticStrategyEngine
        import yaml

        config_path = Path('config.yaml')
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

        for asset in ['BTC', 'ETH']:
            logger.info(f"Evolving {asset} strategies (pop={population_size})...")
            try:
                engine = GeneticStrategyEngine(config=config, population_size=population_size)
                engine.load_market_data(asset=asset, timeframe='4h')

                # 10 generations x population_size strategies
                hall_of_fame = engine.evolve(
                    generations=10,
                    population_size=population_size,
                )

                if hall_of_fame:
                    best = hall_of_fame[0]
                    live_trades = getattr(best, 'live_trades', 0)
                    live_wins = getattr(best, 'live_wins', 0)
                    win_rate = (live_wins / live_trades * 100) if live_trades > 0 else 0
                    logger.info(
                        f"  {asset} best: {best.name}  fitness={best.fitness:.3f}  "
                        f"pnl={best.total_pnl:+.1f}%  live={live_trades} trades  "
                        f"live_wr={win_rate:.0f}%"
                    )
                else:
                    logger.warning(f"  {asset}: No hall of fame entries")

            except Exception as e:
                logger.error(f"  {asset} evolution failed: {e}", exc_info=True)

    except ImportError as e:
        logger.error(f"Import error: {e}")
    except Exception as e:
        logger.error(f"Evolution cycle failed: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description='Genetic Strategy Evolution Loop')
    parser.add_argument('--population_size', type=int, default=50,
                        help='Number of DNA strategies in population (default: 50)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Hours between evolution cycles (default: 2)')
    parser.add_argument('--once', action='store_true',
                        help='Run one cycle and exit')
    args = parser.parse_args()

    interval_sec = int(args.interval * 3600)

    logger.info("Genetic Evolution Loop starting")
    logger.info(f"  Population size: {args.population_size}")
    logger.info(f"  Interval:        {args.interval}h ({interval_sec}s)")

    while True:
        logger.info("=" * 60)
        logger.info("Starting genetic evolution cycle...")
        run_evolution_cycle(args.population_size)
        logger.info(f"Cycle complete. Next in {args.interval}h.")

        if args.once:
            break

        time.sleep(interval_sec)


if __name__ == '__main__':
    main()
