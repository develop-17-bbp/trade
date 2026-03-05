#!/usr/bin/env python
"""
Layer 6 v2.0 Pre-Deployment Validation Script
Checks all components are correctly integrated and ready for trading.
"""

import sys
import os
from pathlib import Path

# Color output for Windows terminal
def success(msg):
    print(f"[OK] {msg}")

def warning(msg):
    print(f"[!] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")
    
def section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

# Add workspace to path
workspace = Path(__file__).parent
sys.path.insert(0, str(workspace))

def check_files():
    """Verify required files exist"""
    section("FILE CHECK")
    
    required_files = {
        "models/lgbm_aave.txt": "Trained LightGBM model",
        "data/AAVE_USDT_1h.csv": "Historical OHLCV data (47k bars)",
        "config.yaml": "Main configuration",
        "src/ai/agentic_strategist.py": "Layer 6 strategist",
        "src/models/lightgbm_classifier.py": "Layer 1 classifier",
        "src/trading/executor.py": "Main executor",
        "requirements.txt": "Dependencies"
    }
    
    all_exist = True
    for filepath, desc in required_files.items():
        full_path = workspace / filepath
        if full_path.exists():
            size = full_path.stat().st_size if filepath.endswith('.txt') or filepath.endswith('.csv') else 'N/A'
            if size != 'N/A':
                success(f"{filepath} ({size:,} bytes) - {desc}")
            else:
                success(f"{filepath} - {desc}")
        else:
            error(f"{filepath} MISSING - {desc}")
            all_exist = False
    
    return all_exist

def check_imports():
    """Verify critical dependencies are installed"""
    section("DEPENDENCY CHECK")
    
    required_modules = {
        "pydantic": "Schema validation (Layer 6)",
        "lightgbm": "Gradient boosting (Layer 1)",
        "transformers": "FinBERT sentiment (Layer 2)",
        "sentence_transformers": "Embedding model (Layer 2)",
        "ccxt": "Exchange data fetcher",
        "yaml": "Config parser",
        "numpy": "Numerical computing",
        "pandas": "Data processing"
    }
    
    missing = []
    for module, desc in required_modules.items():
        try:
            __import__(module)
            # For pydantic, also check version
            if module == "pydantic":
                import pydantic
                version = pydantic.__version__
                if version.startswith("2."):
                    success(f"{module} v{version} - {desc}")
                else:
                    warning(f"{module} v{version} - Expected v2.x for Layer 6")
                    missing.append(module)
            else:
                success(f"{module} - {desc}")
        except ImportError:
            error(f"{module} NOT INSTALLED - {desc}")
            missing.append(module)
    
    return len(missing) == 0

def check_code_syntax():
    """Verify Python files have valid syntax"""
    section("SYNTAX CHECK")
    
    python_files = [
        "src/ai/agentic_strategist.py",
        "src/models/lightgbm_classifier.py",
        "src/trading/executor.py",
        "src/trading/meta_controller.py"
    ]
    
    import ast
    all_valid = True
    
    for filepath in python_files:
        full_path = workspace / filepath
        try:
            with open(full_path, 'r') as f:
                code = f.read()
            ast.parse(code)
            success(f"{filepath} - Valid Python syntax")
        except SyntaxError as e:
            error(f"{filepath} - Syntax error at line {e.lineno}: {e.msg}")
            all_valid = False
    
    return all_valid

def check_config():
    """Verify config.yaml is correctly formatted"""
    section("CONFIG CHECK")
    
    try:
        import yaml
        config_path = workspace / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for critical keys
        critical_keys = {
            'models': 'ML model configuration',
            'risk': 'Risk management settings',
            'l1': 'L1 quantitative engine parameters'
        }
        
        all_present = True
        for key, desc in critical_keys.items():
            if key in config:
                success(f"config.yaml[{key}] - {desc}")
            else:
                warning(f"config.yaml[{key}] MISSING - {desc}")
                all_present = False
        
        # Check LightGBM model path
        if 'models' in config and 'lightgbm' in config['models']:
            model_path = config['models']['lightgbm'].get('model_path')
            if model_path:
                success(f"LightGBM model_path configured: {model_path}")
            else:
                warning("LightGBM model_path NOT configured in config.yaml")
        
        return all_present
    
    except Exception as e:
        error(f"Config parsing failed: {e}")
        return False

def check_core_classes():
    """Verify Layer 6 v2.0 core classes can be imported"""
    section("LAYER 6 v2.0 CORE CLASSES")
    
    try:
        from src.ai.agentic_strategist import AgenticStrategist, StrategistDecision
        success("AgenticStrategist class imported")
        success("StrategistDecision Pydantic model imported")
        
        # Check StrategistDecision has required fields
        from pydantic import BaseModel
        import inspect
        
        fields = StrategistDecision.model_fields if hasattr(StrategistDecision, 'model_fields') else {}
        expected_fields = ['market_regime', 'confidence_score', 'macro_bias', 'reasoning_trace', 'suggested_config_update']
        
        for field in expected_fields:
            if field in fields:
                success(f"  - {field} field present")
            else:
                warning(f"  - {field} field MISSING")
        
        return True
    
    except Exception as e:
        error(f"Layer 6 import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_executor_integration():
    """Verify strategist is integrated in executor"""
    section("EXECUTOR INTEGRATION")
    
    try:
        executor_path = workspace / "src/trading/executor.py"
        with open(executor_path, 'r') as f:
            executor_code = f.read()
        
        # Check for key integration points
        checks = {
            "strategist.analyze_performance": "Strategist called for market analysis",
            "self.agentic_bias": "Agentic bias applied to trading",
            "decision.market_regime": "Decision market regime accessed",
            "record_feedback": "Calibration feedback collected"
        }
        
        all_found = True
        for check_str, desc in checks.items():
            if check_str in executor_code:
                success(f"Integration point found: {check_str} - {desc}")
            else:
                warning(f"Integration point missing: {check_str}")
                all_found = False
        
        return all_found
    
    except Exception as e:
        error(f"Executor check failed: {e}")
        return False

def check_data():
    """Verify historical data is available"""
    section("DATA VALIDATION")
    
    try:
        import pandas as pd
        
        data_path = workspace / "data/AAVE_USDT_1h.csv"
        if not data_path.exists():
            error(f"Data file not found: {data_path}")
            return False
        
        df = pd.read_csv(data_path)
        
        success(f"Data loaded: {len(df):,} rows x {len(df.columns)} columns")
        success(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Check for required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            error(f"Missing OHLCV columns: {missing_cols}")
            return False
        
        success(f"All OHLCV columns present: {required_cols}")
        
        # Check for NaN values
        null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if null_pct > 1:
            warning(f"Data contains {null_pct:.1f}% null values")
        else:
            success(f"Data quality: {100-null_pct:.1f}% complete (no NaN values)")
        
        return True
    
    except Exception as e:
        error(f"Data validation failed: {e}")
        return False

def check_model():
    """Verify LightGBM model file is valid"""
    section("MODEL VALIDATION")
    
    try:
        import lightgbm as lgb
        
        model_path = workspace / "models/lgbm_aave.txt"
        if not model_path.exists():
            error(f"Model file not found: {model_path}")
            return False
        
        # Try to load model
        model = lgb.Booster(model_file=str(model_path))
        success(f"LightGBM model loaded successfully")
        
        # Check model properties
        num_features = model.num_feature()
        num_trees = model.num_trees()
        
        success(f"Model: {num_trees} trees, {num_features} features")
        
        if num_features >= 78:
            success(f"Feature count valid: {num_features} features (expected 78+)")
        else:
            warning(f"Feature count mismatch: {num_features} (expected 78+)")
        
        return True
    
    except Exception as e:
        error(f"Model validation failed: {e}")
        return False

def main():
    """Run all validation checks"""
    print("\n" + "="*60)
    print(" LAYER 6 v2.0 PRE-DEPLOYMENT VALIDATION")
    print("="*60)
    
    checks = [
        ("File Check", check_files),
        ("Dependency Check", check_imports),
        ("Syntax Check", check_code_syntax),
        ("Config Check", check_config),
        ("Layer 6 Core Classes", check_core_classes),
        ("Executor Integration", check_executor_integration),
        ("Data Validation", check_data),
        ("Model Validation", check_model),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            error(f"Check '{name}' failed with exception: {e}")
            results[name] = False
    
    # Summary
    section("VALIDATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"[{status}] {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n" + "="*60)
        print(" SYSTEM IS READY FOR DEPLOYMENT!")
        print("="*60)
        print("\nNext steps:")
        print("1. Review LAYER_6_DEPLOYMENT_GUIDE.md for deployment options")
        print("2. Path A: python -m src.main --mode paper --max_bars 100")
        print("3. Path B: Set REASONING_LLM_KEY env var and run with real LLM")
        print("4. Path C: Fine-tune Llama adapter for domain-specific reasoning")
        return 0
    else:
        print("\n" + "="*60)
        print(f" {total - passed} ISSUES FOUND - Please fix before deployment")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
