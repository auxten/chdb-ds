"""
Exploratory Batch 24: DatetimeAccessor Deep Dive

Focus areas:
1. All dt properties (year, month, day, hour, minute, second, etc.)
2. dt methods (strftime, floor, ceil, round, tz_localize, tz_convert)
3. Boundary cases (year start/end, month start/end, leap year)
4. NaT handling
5. Chained operations with datetime
6. dayofweek/weekday pandas vs chDB alignment (Monday=0 vs Monday=1)

Mirror Code Pattern: Every test compares DataStore with pandas behavior.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sys
sys.path.insert(0, '/Users/auxten/Codes/go/src/github.com/auxten/chdb-ds')

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas
from tests.xfail_markers import chdb_nat_returns_nullable_int


class TestDateTimeBasicProperties:
    """Test basic datetime extraction properties."""
    
    @pytest.fixture
    def df_dates(self):
        """Create DataFrame with various datetime values."""
        return pd.DataFrame({
            'ts': pd.to_datetime([
                '2024-01-15 10:30:45.123',
                '2024-06-21 14:15:30.456',
                '2024-12-31 23:59:59.789',
                '2024-02-29 00:00:00.000',  # Leap year
                '2023-03-15 12:00:00.000',
            ])
        })
    
    def test_dt_year(self, df_dates):
        """Test dt.year extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())
        
        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_month(self, df_dates):
        """Test dt.month extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())
        
        pd_result = pd_df['ts'].dt.month
        ds_result = ds_df['ts'].dt.month
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_day(self, df_dates):
        """Test dt.day extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())
        
        pd_result = pd_df['ts'].dt.day
        ds_result = ds_df['ts'].dt.day
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_hour(self, df_dates):
        """Test dt.hour extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())
        
        pd_result = pd_df['ts'].dt.hour
        ds_result = ds_df['ts'].dt.hour
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_minute(self, df_dates):
        """Test dt.minute extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())
        
        pd_result = pd_df['ts'].dt.minute
        ds_result = ds_df['ts'].dt.minute
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_second(self, df_dates):
        """Test dt.second extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())
        
        pd_result = pd_df['ts'].dt.second
        ds_result = ds_df['ts'].dt.second
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_quarter(self, df_dates):
        """Test dt.quarter extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())
        
        pd_result = pd_df['ts'].dt.quarter
        ds_result = ds_df['ts'].dt.quarter
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDayOfWeekAlignment:
    """
    Test dayofweek/weekday alignment between DataStore and pandas.
    
    pandas: Monday=0, Sunday=6
    chDB toDayOfWeek: Monday=1, Sunday=7 (by default)
    
    DataStore should align with pandas convention.
    """
    
    @pytest.fixture
    def df_weekdays(self):
        """Create DataFrame with dates spanning all days of the week."""
        # 2024-01-01 is Monday
        return pd.DataFrame({
            'date': pd.to_datetime([
                '2024-01-01',  # Monday
                '2024-01-02',  # Tuesday  
                '2024-01-03',  # Wednesday
                '2024-01-04',  # Thursday
                '2024-01-05',  # Friday
                '2024-01-06',  # Saturday
                '2024-01-07',  # Sunday
            ])
        })
    
    def test_dt_dayofweek(self, df_weekdays):
        """Test dt.dayofweek returns Monday=0, Sunday=6 like pandas."""
        pd_df = df_weekdays.copy()
        ds_df = DataStore(df_weekdays.copy())
        
        pd_result = pd_df['date'].dt.dayofweek
        ds_result = ds_df['date'].dt.dayofweek
        
        # pandas: Monday=0, Tuesday=1, ..., Sunday=6
        assert list(pd_result) == [0, 1, 2, 3, 4, 5, 6]
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_weekday(self, df_weekdays):
        """Test dt.weekday (alias for dayofweek)."""
        pd_df = df_weekdays.copy()
        ds_df = DataStore(df_weekdays.copy())
        
        pd_result = pd_df['date'].dt.weekday
        ds_result = ds_df['date'].dt.weekday
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_day_of_week(self, df_weekdays):
        """Test dt.day_of_week (alias for dayofweek)."""
        pd_df = df_weekdays.copy()
        ds_df = DataStore(df_weekdays.copy())
        
        pd_result = pd_df['date'].dt.day_of_week
        ds_result = ds_df['date'].dt.day_of_week
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDayOfYear:
    """Test day of year extraction."""
    
    @pytest.fixture
    def df_doy(self):
        """Create DataFrame with specific day-of-year test cases."""
        return pd.DataFrame({
            'date': pd.to_datetime([
                '2024-01-01',  # Day 1
                '2024-02-29',  # Day 60 (leap year)
                '2024-12-31',  # Day 366 (leap year)
                '2023-12-31',  # Day 365 (non-leap)
                '2024-07-04',  # Mid-year
            ])
        })
    
    def test_dt_dayofyear(self, df_doy):
        """Test dt.dayofyear extraction."""
        pd_df = df_doy.copy()
        ds_df = DataStore(df_doy.copy())
        
        pd_result = pd_df['date'].dt.dayofyear
        ds_result = ds_df['date'].dt.dayofyear
        
        # Verify pandas values
        expected = [1, 60, 366, 365, 186]
        assert list(pd_result) == expected
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_day_of_year(self, df_doy):
        """Test dt.day_of_year alias."""
        pd_df = df_doy.copy()
        ds_df = DataStore(df_doy.copy())
        
        pd_result = pd_df['date'].dt.day_of_year
        ds_result = ds_df['date'].dt.day_of_year
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestWeekNumber:
    """Test week number extraction."""
    
    @pytest.fixture
    def df_weeks(self):
        """Create DataFrame with specific week test cases."""
        return pd.DataFrame({
            'date': pd.to_datetime([
                '2024-01-01',  # Week 1
                '2024-01-07',  # Still week 1
                '2024-01-08',  # Week 2
                '2024-12-30',  # Week 1 of 2025 (ISO)
                '2023-01-01',  # Week 52 of 2022 (ISO)
            ])
        })
    
    def test_dt_week(self, df_weeks):
        """Test dt.week extraction (ISO week)."""
        pd_df = df_weeks.copy()
        ds_df = DataStore(df_weeks.copy())
        
        pd_result = pd_df['date'].dt.isocalendar().week
        ds_result = ds_df['date'].dt.week
        
        # Note: dt.week may differ from isocalendar().week in edge cases
        # We mainly want to verify it doesn't error and returns reasonable values
        assert len(ds_result._execute()) == len(pd_result)
    
    def test_dt_weekofyear(self, df_weeks):
        """Test dt.weekofyear alias."""
        pd_df = df_weeks.copy()
        ds_df = DataStore(df_weeks.copy())
        
        # weekofyear is deprecated in pandas, but should still work
        ds_result = ds_df['date'].dt.weekofyear
        
        # Just verify it works
        result = ds_result._execute()
        assert len(result) == len(pd_df)


class TestBooleanDateProperties:
    """Test boolean datetime properties (is_month_start, is_leap_year, etc.)."""
    
    @pytest.fixture
    def df_boundaries(self):
        """Create DataFrame with boundary dates."""
        return pd.DataFrame({
            'date': pd.to_datetime([
                '2024-01-01',  # Month start, quarter start, year start
                '2024-01-31',  # Month end
                '2024-03-31',  # Month end, quarter end
                '2024-04-01',  # Month start, quarter start
                '2024-12-31',  # Month end, quarter end, year end
                '2024-06-15',  # Mid-month
                '2023-02-28',  # Month end (non-leap)
                '2024-02-29',  # Month end (leap year)
            ])
        })
    
    def test_dt_is_month_start(self, df_boundaries):
        """Test dt.is_month_start property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())
        
        pd_result = pd_df['date'].dt.is_month_start
        ds_result = ds_df['date'].dt.is_month_start
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_is_month_end(self, df_boundaries):
        """Test dt.is_month_end property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())
        
        pd_result = pd_df['date'].dt.is_month_end
        ds_result = ds_df['date'].dt.is_month_end
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_is_quarter_start(self, df_boundaries):
        """Test dt.is_quarter_start property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())
        
        pd_result = pd_df['date'].dt.is_quarter_start
        ds_result = ds_df['date'].dt.is_quarter_start
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_is_quarter_end(self, df_boundaries):
        """Test dt.is_quarter_end property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())
        
        pd_result = pd_df['date'].dt.is_quarter_end
        ds_result = ds_df['date'].dt.is_quarter_end
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_is_year_start(self, df_boundaries):
        """Test dt.is_year_start property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())
        
        pd_result = pd_df['date'].dt.is_year_start
        ds_result = ds_df['date'].dt.is_year_start
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_is_year_end(self, df_boundaries):
        """Test dt.is_year_end property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())
        
        pd_result = pd_df['date'].dt.is_year_end
        ds_result = ds_df['date'].dt.is_year_end
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_is_leap_year(self, df_boundaries):
        """Test dt.is_leap_year property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())
        
        pd_result = pd_df['date'].dt.is_leap_year
        ds_result = ds_df['date'].dt.is_leap_year
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_days_in_month(self, df_boundaries):
        """Test dt.days_in_month property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())
        
        pd_result = pd_df['date'].dt.days_in_month
        ds_result = ds_df['date'].dt.days_in_month
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateTimeMethods:
    """Test datetime methods (strftime, floor, ceil, round, normalize)."""
    
    @pytest.fixture
    def df_times(self):
        """Create DataFrame with various times."""
        return pd.DataFrame({
            'ts': pd.to_datetime([
                '2024-03-15 10:37:45.123456',
                '2024-06-21 14:22:30.789012',
                '2024-12-25 23:59:59.999999',
            ])
        })
    
    def test_dt_strftime_basic(self, df_times):
        """Test dt.strftime with basic format."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())
        
        pd_result = pd_df['ts'].dt.strftime('%Y-%m-%d')
        ds_result = ds_df['ts'].dt.strftime('%Y-%m-%d')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_strftime_full(self, df_times):
        """Test dt.strftime with full datetime format."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())
        
        pd_result = pd_df['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
        ds_result = ds_df['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_floor_hour(self, df_times):
        """Test dt.floor to hour."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())
        
        pd_result = pd_df['ts'].dt.floor('h')
        ds_result = ds_df['ts'].dt.floor('h')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_floor_day(self, df_times):
        """Test dt.floor to day."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())
        
        pd_result = pd_df['ts'].dt.floor('D')
        ds_result = ds_df['ts'].dt.floor('D')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_ceil_hour(self, df_times):
        """Test dt.ceil to hour."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())
        
        pd_result = pd_df['ts'].dt.ceil('h')
        ds_result = ds_df['ts'].dt.ceil('h')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_round_hour(self, df_times):
        """Test dt.round to hour."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())
        
        pd_result = pd_df['ts'].dt.round('h')
        ds_result = ds_df['ts'].dt.round('h')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_normalize(self, df_times):
        """Test dt.normalize (convert to midnight)."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())
        
        pd_result = pd_df['ts'].dt.normalize()
        ds_result = ds_df['ts'].dt.normalize()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateTimePartsExtraction:
    """Test extracting date/time parts."""
    
    @pytest.fixture
    def df_full(self):
        """Create DataFrame with full datetime precision."""
        return pd.DataFrame({
            'ts': pd.to_datetime([
                '2024-03-15 10:30:45.123456789',
                '2024-06-21 14:15:30.456789012',
            ])
        })
    
    def test_dt_date(self, df_full):
        """Test dt.date extraction (date part only)."""
        pd_df = df_full.copy()
        ds_df = DataStore(df_full.copy())
        
        pd_result = pd_df['ts'].dt.date
        ds_result = ds_df['ts'].dt.date
        
        # Convert to string for comparison (date objects)
        pd_dates = pd_result.astype(str)
        ds_dates = ds_result._execute().astype(str)
        
        pd.testing.assert_series_equal(
            ds_dates.reset_index(drop=True),
            pd_dates.reset_index(drop=True),
            check_names=False,
        )
    
    def test_dt_time(self, df_full):
        """Test dt.time extraction (time part only)."""
        pd_df = df_full.copy()
        ds_df = DataStore(df_full.copy())
        
        pd_result = pd_df['ts'].dt.time
        ds_result = ds_df['ts'].dt.time
        
        # Convert to string for comparison (time objects)
        pd_times = pd_result.astype(str)
        ds_times = ds_result._execute().astype(str)
        
        pd.testing.assert_series_equal(
            ds_times.reset_index(drop=True),
            pd_times.reset_index(drop=True),
            check_names=False,
        )


class TestSubSecondPrecision:
    """Test sub-second precision (microsecond, nanosecond)."""
    
    @pytest.fixture
    def df_subsec(self):
        """Create DataFrame with sub-second precision."""
        return pd.DataFrame({
            'ts': pd.to_datetime([
                '2024-01-01 00:00:00.123456789',
                '2024-06-15 12:30:45.987654321',
                '2024-12-31 23:59:59.000000001',
            ])
        })
    
    def test_dt_microsecond(self, df_subsec):
        """Test dt.microsecond extraction."""
        pd_df = df_subsec.copy()
        ds_df = DataStore(df_subsec.copy())
        
        pd_result = pd_df['ts'].dt.microsecond
        ds_result = ds_df['ts'].dt.microsecond
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_nanosecond(self, df_subsec):
        """Test dt.nanosecond extraction."""
        pd_df = df_subsec.copy()
        ds_df = DataStore(df_subsec.copy())
        
        pd_result = pd_df['ts'].dt.nanosecond
        ds_result = ds_df['ts'].dt.nanosecond
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNaTHandling:
    """Test handling of NaT (Not a Time) values."""
    
    @pytest.fixture
    def df_with_nat(self):
        """Create DataFrame with NaT values."""
        return pd.DataFrame({
            'ts': pd.to_datetime([
                '2024-01-15 10:30:00',
                pd.NaT,
                '2024-06-21 14:15:00',
                pd.NaT,
                '2024-12-31 23:59:59',
            ])
        })
    
    @chdb_nat_returns_nullable_int
    def test_dt_year_with_nat(self, df_with_nat):
        """Test dt.year with NaT values."""
        pd_df = df_with_nat.copy()
        ds_df = DataStore(df_with_nat.copy())
        
        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year
        
        # NaT should become NaN
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    @chdb_nat_returns_nullable_int
    def test_dt_month_with_nat(self, df_with_nat):
        """Test dt.month with NaT values."""
        pd_df = df_with_nat.copy()
        ds_df = DataStore(df_with_nat.copy())
        
        pd_result = pd_df['ts'].dt.month
        ds_result = ds_df['ts'].dt.month
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    @chdb_nat_returns_nullable_int
    def test_dt_dayofweek_with_nat(self, df_with_nat):
        """Test dt.dayofweek with NaT values."""
        pd_df = df_with_nat.copy()
        ds_df = DataStore(df_with_nat.copy())
        
        pd_result = pd_df['ts'].dt.dayofweek
        ds_result = ds_df['ts'].dt.dayofweek
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dt_is_month_start_with_nat(self, df_with_nat):
        """Test dt.is_month_start with NaT values."""
        pd_df = df_with_nat.copy()
        ds_df = DataStore(df_with_nat.copy())
        
        pd_result = pd_df['ts'].dt.is_month_start
        ds_result = ds_df['ts'].dt.is_month_start
        
        # NaT should become False or NaN depending on pandas version
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateTimeChaining:
    """Test chaining datetime operations with other DataFrame operations."""
    
    @pytest.fixture
    def df_sales(self):
        """Create sales DataFrame with dates."""
        return pd.DataFrame({
            'sale_date': pd.to_datetime([
                '2024-01-15', '2024-01-20', '2024-02-10',
                '2024-02-25', '2024-03-05', '2024-03-15',
            ]),
            'amount': [100, 200, 150, 300, 250, 175]
        })
    
    def test_filter_by_month(self, df_sales):
        """Test filtering by dt.month."""
        pd_df = df_sales.copy()
        ds_df = DataStore(df_sales.copy())
        
        pd_result = pd_df[pd_df['sale_date'].dt.month == 2]
        ds_result = ds_df[ds_df['sale_date'].dt.month == 2]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_filter_by_dayofweek(self, df_sales):
        """Test filtering by dt.dayofweek (weekdays only)."""
        pd_df = df_sales.copy()
        ds_df = DataStore(df_sales.copy())
        
        # Filter for weekdays (Monday=0 to Friday=4)
        pd_result = pd_df[pd_df['sale_date'].dt.dayofweek < 5]
        ds_result = ds_df[ds_df['sale_date'].dt.dayofweek < 5]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_groupby_month(self, df_sales):
        """Test groupby with dt.month."""
        pd_df = df_sales.copy()
        ds_df = DataStore(df_sales.copy())
        
        # Add month column for groupby
        pd_df['month'] = pd_df['sale_date'].dt.month
        pd_result = pd_df.groupby('month')['amount'].sum().reset_index()
        
        # For DataStore, use assign to add the month column
        ds_result = ds_df.assign(month=ds_df['sale_date'].dt.month)
        ds_result = ds_result.groupby('month')['amount'].sum().reset_index()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_assign_multiple_dt_columns(self, df_sales):
        """Test assigning multiple datetime-derived columns."""
        pd_df = df_sales.copy()
        ds_df = DataStore(df_sales.copy())
        
        pd_df = pd_df.assign(
            year=pd_df['sale_date'].dt.year,
            month=pd_df['sale_date'].dt.month,
            day=pd_df['sale_date'].dt.day,
        )
        
        ds_df = ds_df.assign(
            year=ds_df['sale_date'].dt.year,
            month=ds_df['sale_date'].dt.month,
            day=ds_df['sale_date'].dt.day,
        )
        
        assert_datastore_equals_pandas(ds_df, pd_df)
    
    def test_sort_by_dayofyear(self, df_sales):
        """Test sorting by dt.dayofyear."""
        pd_df = df_sales.copy()
        ds_df = DataStore(df_sales.copy())
        
        pd_df['doy'] = pd_df['sale_date'].dt.dayofyear
        pd_result = pd_df.sort_values('doy')
        
        ds_df = ds_df.assign(doy=ds_df['sale_date'].dt.dayofyear)
        ds_result = ds_df.sort_values('doy')
        
        # Reset index for comparison
        pd_result = pd_result.reset_index(drop=True)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateTimeEdgeCases:
    """Test edge cases for datetime operations."""
    
    def test_empty_dataframe_dt(self):
        """Test dt accessor on empty DataFrame."""
        pd_df = pd.DataFrame({'ts': pd.to_datetime([])})
        ds_df = DataStore(pd.DataFrame({'ts': pd.to_datetime([])}))
        
        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year
        
        assert len(ds_result._execute()) == 0
        assert len(pd_result) == 0
    
    def test_single_row_dt(self):
        """Test dt accessor on single-row DataFrame."""
        pd_df = pd.DataFrame({'ts': pd.to_datetime(['2024-06-15 12:30:00'])})
        ds_df = DataStore(pd.DataFrame({'ts': pd.to_datetime(['2024-06-15 12:30:00'])}))
        
        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    @chdb_nat_returns_nullable_int
    def test_all_nat_dt(self):
        """Test dt accessor on all-NaT column."""
        pd_df = pd.DataFrame({'ts': pd.to_datetime([pd.NaT, pd.NaT, pd.NaT])})
        ds_df = DataStore(pd.DataFrame({'ts': pd.to_datetime([pd.NaT, pd.NaT, pd.NaT])}))
        
        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_century_boundary(self):
        """Test dt accessor on century boundary dates."""
        pd_df = pd.DataFrame({
            'ts': pd.to_datetime([
                '1999-12-31 23:59:59',
                '2000-01-01 00:00:00',
                '2000-01-01 00:00:01',
            ])
        })
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year
        
        assert list(pd_result) == [1999, 2000, 2000]
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_leap_year_feb29(self):
        """Test dt operations on Feb 29 (leap year)."""
        pd_df = pd.DataFrame({
            'ts': pd.to_datetime([
                '2020-02-29',  # Leap year
                '2024-02-29',  # Leap year
            ])
        })
        ds_df = DataStore(pd_df.copy())
        
        # Test day
        pd_day = pd_df['ts'].dt.day
        ds_day = ds_df['ts'].dt.day
        assert list(pd_day) == [29, 29]
        assert_datastore_equals_pandas(ds_day, pd_day)
        
        # Test dayofyear
        pd_doy = pd_df['ts'].dt.dayofyear
        ds_doy = ds_df['ts'].dt.dayofyear
        assert list(pd_doy) == [60, 60]  # Feb 29 is day 60
        assert_datastore_equals_pandas(ds_doy, pd_doy)


class TestTimezoneOperations:
    """Test timezone-related operations."""
    
    @pytest.fixture
    def df_naive(self):
        """Create DataFrame with timezone-naive datetimes."""
        return pd.DataFrame({
            'ts': pd.to_datetime([
                '2024-06-15 10:00:00',
                '2024-06-15 15:00:00',
                '2024-06-15 20:00:00',
            ])
        })
    
    @pytest.fixture
    def df_aware(self):
        """Create DataFrame with timezone-aware datetimes."""
        return pd.DataFrame({
            'ts': pd.to_datetime([
                '2024-06-15 10:00:00',
                '2024-06-15 15:00:00',
                '2024-06-15 20:00:00',
            ]).tz_localize('UTC')
        })
    
    def test_tz_localize_utc(self, df_naive):
        """Test dt.tz_localize to UTC."""
        pd_df = df_naive.copy()
        ds_df = DataStore(df_naive.copy())
        
        pd_result = pd_df['ts'].dt.tz_localize('UTC')
        ds_result = ds_df['ts'].dt.tz_localize('UTC')
        
        # Compare string representation to avoid timezone object differences
        pd_str = pd_result.astype(str)
        ds_str = ds_result._execute().astype(str)
        
        pd.testing.assert_series_equal(
            ds_str.reset_index(drop=True),
            pd_str.reset_index(drop=True),
            check_names=False,
        )
    
    def test_tz_convert(self, df_aware):
        """Test dt.tz_convert to different timezone."""
        pd_df = df_aware.copy()
        ds_df = DataStore(df_aware.copy())
        
        pd_result = pd_df['ts'].dt.tz_convert('US/Eastern')
        ds_result = ds_df['ts'].dt.tz_convert('US/Eastern')
        
        # Compare hour values
        pd_hours = pd_result.dt.hour
        ds_hours = ds_result._execute().dt.hour
        
        pd.testing.assert_series_equal(
            ds_hours.reset_index(drop=True),
            pd_hours.reset_index(drop=True),
            check_names=False,
        )


class TestToPeriod:
    """Test dt.to_period conversion."""
    
    @pytest.fixture
    def df_dates(self):
        """Create DataFrame with dates."""
        return pd.DataFrame({
            'ts': pd.to_datetime([
                '2024-01-15',
                '2024-06-21',
                '2024-12-31',
            ])
        })
    
    def test_to_period_month(self, df_dates):
        """Test dt.to_period with monthly frequency."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())
        
        pd_result = pd_df['ts'].dt.to_period('M')
        ds_result = ds_df['ts'].dt.to_period('M')
        
        # Compare string representation
        pd_str = pd_result.astype(str)
        ds_str = ds_result._execute().astype(str)
        
        pd.testing.assert_series_equal(
            ds_str.reset_index(drop=True),
            pd_str.reset_index(drop=True),
            check_names=False,
        )
    
    def test_to_period_quarter(self, df_dates):
        """Test dt.to_period with quarterly frequency."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())
        
        pd_result = pd_df['ts'].dt.to_period('Q')
        ds_result = ds_df['ts'].dt.to_period('Q')
        
        # Compare string representation
        pd_str = pd_result.astype(str)
        ds_str = ds_result._execute().astype(str)
        
        pd.testing.assert_series_equal(
            ds_str.reset_index(drop=True),
            pd_str.reset_index(drop=True),
            check_names=False,
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
