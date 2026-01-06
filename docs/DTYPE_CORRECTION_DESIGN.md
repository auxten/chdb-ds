# Dtype Correction Architecture Design

## Problem Statement

chDB and pandas have different dtype behaviors for various numeric operations. This causes mismatches that need to be corrected for pandas compatibility.

## Research Findings

### Categories of Dtype Mismatches

| Category | Count | Description | Priority |
|----------|-------|-------------|----------|
| Sign Change | 4 | abs() on signed int → unsigned int | **CRITICAL** |
| Family Change | 13 | int ↔ float type conversion | HIGH |
| Type Widening | 26 | chDB uses wider int type | MEDIUM |
| Type Narrowing | 6 | chDB optimizes to smaller type | MEDIUM |
| Precision Change | 9 | float32 → float64 | LOW |

### Detailed Findings

**1. Sign Change (CRITICAL)**
- `abs(int8/16/32/64)` → returns `uint8/16/32/64` (should be `int8/16/32/64`)

**2. Family Change (HIGH)**
- `pow(int, n)` → returns `float64` (should be `int` for integer powers)
- `sign(float)` → returns `int8` (should be `float`)
- `floordiv(float, n)` → returns `int` (should be `float`)

**3. Type Widening/Narrowing (MEDIUM)**
- `add/sub/mul(int8, scalar)` → chDB widens to prevent overflow
- `mod(int64, n)` → chDB narrows to smallest sufficient type
- `sign(int)` → always `int8`

**4. Precision Change (LOW)**
- `op(float32, scalar)` → chDB promotes to `float64`

## Design Principles

1. **Input-aware correction**: Corrections depend on input dtype
2. **Configurable**: Allow users to enable/disable corrections
3. **Extensible**: Easy to add new correction rules
4. **Centralized**: Single source of truth for dtype rules
5. **Minimal overhead**: Only apply corrections when necessary

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    DtypeCorrectionRegistry                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Rules: Dict[str, DtypeCorrectionRule]                   │   │
│  │  - "abs" → AbsCorrectionRule                             │   │
│  │  - "sign" → SignCorrectionRule                           │   │
│  │  - "pow" → PowCorrectionRule                             │   │
│  │  - ...                                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Methods:                                                        │
│  - get_rule(func_name) → DtypeCorrectionRule                    │
│  - should_correct(func_name, input_dtype, output_dtype) → bool  │
│  - get_target_dtype(func_name, input_dtype) → str               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DtypeCorrectionRule (Abstract)                │
│                                                                  │
│  Properties:                                                     │
│  - func_name: str                                                │
│  - priority: CorrectionPriority (CRITICAL, HIGH, MEDIUM, LOW)   │
│                                                                  │
│  Methods:                                                        │
│  - should_correct(input_dtype, output_dtype) → bool             │
│  - get_target_dtype(input_dtype) → str                          │
│  - apply(series, input_dtype) → Series                          │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ SignedAbs   │    │ PreserveType│    │ SignPreserve│
   │ Rule        │    │ Rule        │    │ Rule        │
   └─────────────┘    └─────────────┘    └─────────────┘
```

### Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                        SQL Execution Path                        │
│                                                                  │
│  1. Before SQL: Capture input dtypes for columns used in exprs  │
│  2. Execute SQL                                                  │
│  3. After SQL: Apply dtype corrections via registry              │
│                                                                  │
│  SQLExecutionEngine.execute_sql_on_dataframe()                  │
│    └── _apply_dtype_corrections(result_df, input_df, plan)      │
│          └── registry.apply_corrections(...)                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Expression Evaluation Path                   │
│                                                                  │
│  ExpressionEvaluator._evaluate_via_chdb()                       │
│    └── _apply_dtype_correction(expr, result, input_col)         │
│          └── registry.apply_corrections(...)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Correction Rule Definitions

```python
# Priority levels
class CorrectionPriority(Enum):
    CRITICAL = 1  # Sign changes - must fix
    HIGH = 2      # Family changes - should fix
    MEDIUM = 3    # Type width changes - nice to fix
    LOW = 4       # Precision changes - optional

# Rule configurations
CORRECTION_RULES = {
    # Sign Change (CRITICAL)
    "abs": {
        "priority": CRITICAL,
        "rule": "unsigned_to_signed",
        "mapping": {
            ("int8", "uint8"): "int8",
            ("int16", "uint16"): "int16",
            ("int32", "uint32"): "int32",
            ("int64", "uint64"): "int64",
        }
    },
    
    # Family Change (HIGH) 
    "sign": {
        "priority": HIGH,
        "rule": "preserve_input_type",
        # input int/float → output should match input family
    },
    
    "pow": {
        "priority": HIGH,
        "rule": "preserve_int_for_int_power",
        # int ** int_literal → should be int (if no overflow)
    },
    
    # Type Width (MEDIUM)
    "add": {
        "priority": MEDIUM,
        "rule": "preserve_input_type",
    },
    "sub": {...},
    "mul": {...},
    "mod": {...},
}
```

## Implementation Plan

### Step 1: Create Core Classes
- `DtypeCorrectionRule` abstract base class
- `DtypeCorrectionRegistry` singleton
- Concrete rule implementations

### Step 2: Define Correction Rules
- Start with CRITICAL (abs) - already done
- Add HIGH priority rules (sign, pow, floordiv)
- Add MEDIUM priority rules (arithmetic ops)

### Step 3: Integration
- Update `SQLExecutionEngine._apply_sql_dtype_corrections()`
- Update `ExpressionEvaluator._apply_dtype_correction()`
- Use registry instead of hardcoded logic

### Step 4: Configuration
- Allow users to enable/disable correction levels
- Config: `dtype_correction_level = "critical" | "high" | "medium" | "all" | "none"`

## File Structure

```
datastore/
├── dtype_correction/
│   ├── __init__.py          # Exports registry and rules
│   ├── registry.py          # DtypeCorrectionRegistry
│   ├── rules.py             # DtypeCorrectionRule base and implementations
│   └── config.py            # Correction configuration
├── expression_evaluator.py  # Use registry
└── sql_executor.py          # Use registry
```

## Testing Strategy

1. Unit tests for each correction rule
2. Integration tests for SQL and Expression paths
3. Comprehensive dtype matrix tests
4. Regression tests for existing functionality

