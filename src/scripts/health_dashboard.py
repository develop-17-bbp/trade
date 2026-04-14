"""
ACT System Health Dashboard — Aggregates All Health Reports
=============================================================
Reads individual health check reports and produces a single
dashboard summary with RED/GREEN/YELLOW status per subsystem.

Usage:
    python -m src.scripts.health_dashboard              # Run all checks + aggregate
    python -m src.scripts.health_dashboard --aggregate   # Aggregate existing reports only
"""

import os
import sys
import json
import time
import importlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('health_dashboard')

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPORTS_DIR = PROJECT_ROOT / 'reports'


def check_apis() -> Dict[str, Any]:
    """Check API/exchange connectivity."""
    result = {'status': 'GREEN', 'apis': {}}

    # Kraken (public, no auth)
    try:
        import ccxt
        exchange = ccxt.kraken({'enableRateLimit': True})
        start = time.time()
        ticker = exchange.fetch_ticker('BTC/USD')
        latency = (time.time() - start) * 1000
        result['apis']['kraken'] = {
            'status': 'GREEN', 'latency_ms': round(latency, 1),
            'details': f"BTC/USD = ${ticker['last']:,.0f}",
        }
    except Exception as e:
        result['apis']['kraken'] = {'status': 'RED', 'error': str(e)}
        result['status'] = 'YELLOW'

    # Robinhood SDK
    try:
        import robin_stocks
        ver = getattr(robin_stocks, '__version__', 'installed')
        result['apis']['robinhood_sdk'] = {
            'status': 'GREEN', 'details': f'robin_stocks {ver}',
        }
    except ImportError:
        result['apis']['robinhood_sdk'] = {
            'status': 'YELLOW', 'details': 'robin_stocks not installed',
        }

    # Ollama endpoints
    import urllib.request, re
    search_files = [
        PROJECT_ROOT / 'src' / 'ai' / 'agentic_strategist.py',
        PROJECT_ROOT / 'src' / 'ai' / 'llm_provider.py',
    ]
    url_pattern = re.compile(r'https?://[a-zA-Z0-9\-]+\.trycloudflare\.com')
    ollama_urls = {}
    for fpath in search_files:
        if fpath.exists():
            try:
                for m in url_pattern.findall(fpath.read_text(errors='ignore')):
                    if m not in ollama_urls.values():
                        ollama_urls[f'ollama_{len(ollama_urls)}'] = m
            except Exception:
                pass

    for name, url in ollama_urls.items():
        try:
            start = time.time()
            req = urllib.request.Request(url, method='GET')
            req.add_header('User-Agent', 'ACT/1.0')
            resp = urllib.request.urlopen(req, timeout=10)
            latency = (time.time() - start) * 1000
            result['apis'][name] = {
                'status': 'GREEN', 'latency_ms': round(latency, 1),
                'url': url,
            }
        except Exception as e:
            result['apis'][name] = {'status': 'RED', 'url': url, 'error': str(e)}
            result['status'] = 'YELLOW'

    return result


def check_models() -> Dict[str, Any]:
    """Check ML model files and loading."""
    result = {'status': 'GREEN', 'models': {}, 'summary': {}}
    models_dir = PROJECT_ROOT / 'models'

    # Find production models
    production_models = {}
    for asset in ['btc', 'eth']:
        for suffix in ['', '_trained']:
            fpath = models_dir / f'lgbm_{asset}{suffix}.txt'
            if fpath.exists():
                production_models[f'lgbm_{asset}{suffix}'] = fpath

    total = len(list(models_dir.glob('*.txt')))
    loadable = 0
    prod_ready = 0

    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        result['status'] = 'RED'
        result['summary'] = {'error': 'lightgbm not installed'}
        return result

    for name, fpath in production_models.items():
        model_info = {
            'file_size_kb': round(os.path.getsize(fpath) / 1024, 1),
            'last_modified': datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat(),
        }
        try:
            model = lgb.Booster(model_file=str(fpath))
            model_info['num_trees'] = model.num_trees()
            model_info['num_features'] = model.num_feature()
            model_info['status'] = 'GREEN'
            loadable += 1

            # Sanity prediction
            fake_features = np.random.randn(1, model.num_feature()).astype(np.float32)
            pred = model.predict(fake_features)[0]
            model_info['test_prediction'] = round(float(pred), 4)
            if 0 <= pred <= 1:
                prod_ready += 1
            else:
                model_info['status'] = 'YELLOW'
                model_info['warning'] = f'Prediction {pred:.4f} outside [0,1]'

            # Freshness
            age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(fpath))).days
            if age_days > 30:
                model_info['freshness'] = 'RED'
                model_info['age_days'] = age_days
            elif age_days > 7:
                model_info['freshness'] = 'YELLOW'
                model_info['age_days'] = age_days
            else:
                model_info['freshness'] = 'GREEN'
                model_info['age_days'] = age_days

        except Exception as e:
            model_info['status'] = 'RED'
            model_info['error'] = str(e)

        result['models'][name] = model_info

    result['summary'] = {
        'total_model_files': total,
        'production_models': len(production_models),
        'loadable': loadable,
        'production_ready': prod_ready,
    }

    if prod_ready == 0 and len(production_models) > 0:
        result['status'] = 'RED'
    elif loadable < len(production_models):
        result['status'] = 'YELLOW'

    return result


def check_imports() -> Dict[str, Any]:
    """Check Python imports and config files."""
    result = {'status': 'GREEN', 'imports': {}, 'config_files': {}, 'pip_packages': {}}

    # Key modules to test
    modules = [
        'src.trading.executor', 'src.trading.genetic_strategy_engine',
        'src.trading.multi_strategy_engine', 'src.trading.adaptive_feedback',
        'src.trading.self_evolving_overlay', 'src.trading.optimizer',
        'src.scripts.continuous_adapt', 'src.indicators.indicators',
        'src.models.lightgbm_classifier', 'src.trading.meta_controller',
        'src.ai.agentic_strategist', 'src.data.fetcher',
    ]

    for mod_name in modules:
        try:
            start = time.time()
            importlib.import_module(mod_name)
            elapsed = (time.time() - start) * 1000
            result['imports'][mod_name] = {
                'status': 'GREEN', 'import_time_ms': round(elapsed, 1),
            }
        except Exception as e:
            result['imports'][mod_name] = {
                'status': 'RED', 'error': str(e)[:200],
            }
            result['status'] = 'YELLOW'

    # Config files
    config_files = [
        'data/recommended_weights.json', 'data/adaptive_state.json',
        'data/evolution_state.json', 'logs/genetic_evolution_results.json',
    ]
    for cf in config_files:
        fpath = PROJECT_ROOT / cf
        if fpath.exists():
            try:
                with open(fpath) as f:
                    json.load(f)
                result['config_files'][cf] = {'status': 'GREEN'}
            except Exception as e:
                result['config_files'][cf] = {'status': 'RED', 'error': str(e)[:100]}
        else:
            result['config_files'][cf] = {'status': 'YELLOW', 'error': 'File not found'}

    # Key pip packages
    packages = ['lightgbm', 'ccxt', 'numpy', 'pandas', 'torch', 'robin_stocks', 'optuna']
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, '__version__', 'unknown')
            result['pip_packages'][pkg] = {'installed': True, 'version': str(ver)}
        except ImportError:
            result['pip_packages'][pkg] = {'installed': False}
            if pkg in ('lightgbm', 'numpy', 'pandas'):
                result['status'] = 'YELLOW'

    return result


def aggregate_reports() -> Dict[str, Any]:
    """Aggregate all health reports into a single dashboard."""
    dashboard = {
        'timestamp': datetime.now(tz=timezone.utc).isoformat(),
        'subsystems': {},
        'overall_status': 'GREEN',
        'recommended_fixes': [],
    }

    # Load or run each check
    checks = {
        'apis': ('health_apis.json', check_apis),
        'models': ('health_models.json', check_models),
        'imports': ('health_imports.json', check_imports),
    }

    for name, (filename, check_fn) in checks.items():
        report_path = REPORTS_DIR / filename
        if report_path.exists():
            try:
                with open(report_path) as f:
                    data = json.load(f)
                dashboard['subsystems'][name] = data
                continue
            except Exception:
                pass
        # Run the check directly
        try:
            data = check_fn()
            dashboard['subsystems'][name] = data
            # Save for future reference
            try:
                with open(report_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception:
                pass
        except Exception as e:
            dashboard['subsystems'][name] = {'status': 'RED', 'error': str(e)}

    # Load daily ops report if available
    today = datetime.now().strftime('%Y-%m-%d')
    daily_path = REPORTS_DIR / f'daily-{today}.json'
    if daily_path.exists():
        try:
            with open(daily_path) as f:
                dashboard['subsystems']['daily_ops'] = json.load(f)
        except Exception:
            pass

    # Determine overall status
    statuses = []
    for name, data in dashboard['subsystems'].items():
        s = data.get('overall_status', data.get('status', 'GREEN'))
        statuses.append(s)

    if 'RED' in statuses:
        dashboard['overall_status'] = 'RED'
    elif 'YELLOW' in statuses:
        dashboard['overall_status'] = 'YELLOW'

    # Generate recommended fixes
    for name, data in dashboard['subsystems'].items():
        status = data.get('overall_status', data.get('status', 'GREEN'))
        if status == 'RED':
            dashboard['recommended_fixes'].append({
                'subsystem': name, 'severity': 'HIGH',
                'fix': f'Investigate {name} subsystem — status RED',
            })

        # Specific recommendations
        if name == 'models':
            for model_name, model_data in data.get('models', {}).items():
                if isinstance(model_data, dict):
                    if model_data.get('freshness') == 'RED':
                        dashboard['recommended_fixes'].append({
                            'subsystem': 'models', 'severity': 'MEDIUM',
                            'fix': f'Retrain {model_name} — {model_data.get("age_days", "?")} days old',
                        })

        if name == 'apis':
            for api_name, api_data in data.get('apis', {}).items():
                if isinstance(api_data, dict) and api_data.get('status') == 'RED':
                    dashboard['recommended_fixes'].append({
                        'subsystem': 'apis', 'severity': 'HIGH',
                        'fix': f'Fix {api_name} connection: {api_data.get("error", "unknown")}',
                    })

    # Save dashboard
    dash_path = REPORTS_DIR / 'health_dashboard.json'
    with open(dash_path, 'w') as f:
        json.dump(dashboard, f, indent=2, default=str)

    return dashboard


def print_dashboard(dashboard: Dict):
    """Pretty-print the dashboard summary."""
    print("\n" + "=" * 65)
    print("  ACT SYSTEM HEALTH DASHBOARD")
    print(f"  {dashboard['timestamp']}")
    print("=" * 65)

    overall = dashboard['overall_status']
    icon = {'GREEN': '[OK]', 'YELLOW': '[!!]', 'RED': '[XX]'}.get(overall, '[??]')
    print(f"\n  OVERALL: {icon} {overall}\n")

    for name, data in dashboard.get('subsystems', {}).items():
        status = data.get('overall_status', data.get('status', '?'))
        icon = {'GREEN': '[OK]', 'YELLOW': '[!!]', 'RED': '[XX]'}.get(status, '[??]')
        print(f"  {icon} {name.upper():20s} {status}")

    fixes = dashboard.get('recommended_fixes', [])
    if fixes:
        print(f"\n  RECOMMENDED FIXES ({len(fixes)}):")
        for fix in fixes:
            sev = fix.get('severity', '?')
            print(f"    [{sev}] {fix.get('fix', '?')}")

    print("\n" + "=" * 65)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='ACT Health Dashboard')
    parser.add_argument('--aggregate', action='store_true', help='Only aggregate existing reports')
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dashboard = aggregate_reports()
    print_dashboard(dashboard)
    print(f"\n  Full report: {REPORTS_DIR / 'health_dashboard.json'}")


if __name__ == '__main__':
    main()
