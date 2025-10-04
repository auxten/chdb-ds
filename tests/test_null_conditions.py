"""
Test NULL condition operations - extended IS NULL/IS NOT NULL tests

Comprehensive NULL handling with various scenarios and chdb execution.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field


# ========== SQL Generation Tests ==========


class NullBasicTests(unittest.TestCase):
    """Basic NULL tests"""

    def test_isnull_basic(self):
        """Test IS NULL"""
        cond = Field("email").isnull()
        self.assertEqual('"email" IS NULL', cond.to_sql())

    def test_notnull_basic(self):
        """Test IS NOT NULL"""
        cond = Field("email").notnull()
        self.assertEqual('"email" IS NOT NULL', cond.to_sql())

    def test_isnull_in_query(self):
        """Test IS NULL in query"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(ds.email.isnull()).to_sql()
        self.assertEqual('SELECT * FROM "users" WHERE "email" IS NULL', sql)

    def test_notnull_in_query(self):
        """Test IS NOT NULL in query"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(ds.phone.notnull()).to_sql()
        self.assertEqual('SELECT * FROM "users" WHERE "phone" IS NOT NULL', sql)


class NullWithCombinationsTests(unittest.TestCase):
    """Test NULL combined with other conditions"""

    def test_null_and_other_condition(self):
        """Test IS NULL AND another condition"""
        ds = DataStore(table="data")
        sql = ds.select("*").filter(ds.email.isnull() & (ds.status == 'active')).to_sql()
        self.assertIn('IS NULL', sql)
        self.assertIn('AND', sql)

    def test_null_or_other_condition(self):
        """Test IS NULL OR another condition"""
        ds = DataStore(table="data")
        sql = ds.select("*").filter(ds.email.isnull() | ds.phone.isnull()).to_sql()
        self.assertIn('IS NULL', sql)
        self.assertIn('OR', sql)

    def test_not_null_and_in(self):
        """Test NOT NULL combined with IN"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(ds.email.notnull() & ds.status.isin(['active', 'premium'])).to_sql()
        self.assertIn('IS NOT NULL', sql)
        self.assertIn('IN', sql)

    def test_not_null_and_like(self):
        """Test NOT NULL combined with LIKE"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(ds.email.notnull() & ds.email.like('%@company.com')).to_sql()
        self.assertIn('IS NOT NULL', sql)
        self.assertIn('LIKE', sql)


class NullNegationTests(unittest.TestCase):
    """Test NOT operator with NULL conditions"""

    def test_not_isnull(self):
        """Test NOT (IS NULL) - equivalent to IS NOT NULL"""
        ds = DataStore(table="data")
        sql = ds.select("*").filter(~ds.email.isnull()).to_sql()
        self.assertIn('NOT', sql)
        self.assertIn('IS NULL', sql)

    def test_not_notnull(self):
        """Test NOT (IS NOT NULL) - equivalent to IS NULL"""
        ds = DataStore(table="data")
        sql = ds.select("*").filter(~ds.email.notnull()).to_sql()
        self.assertIn('NOT', sql)
        self.assertIn('IS NOT NULL', sql)


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class NullExecutionTests(unittest.TestCase):
    """Test NULL condition execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table with NULL values"""
        cls.init_sql = """
        CREATE TABLE test_null_exec (
            id UInt32,
            name String,
            email Nullable(String),
            phone Nullable(String),
            address Nullable(String),
            status String
        ) ENGINE = Memory;
        
        INSERT INTO test_null_exec VALUES
            (1, 'Alice', 'alice@test.com', '555-1234', 'NYC', 'active'),
            (2, 'Bob', NULL, '555-5678', NULL, 'active'),
            (3, 'Charlie', 'charlie@test.com', NULL, 'LA', 'inactive'),
            (4, 'David', NULL, NULL, NULL, 'pending'),
            (5, 'Eve', 'eve@test.com', '555-9999', 'Chicago', 'active'),
            (6, 'Frank', NULL, '555-3333', 'NYC', 'deleted');
        """

        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session"""
        if hasattr(cls, 'session'):
            cls.session.cleanup()

    def _execute(self, sql):
        """Helper to execute SQL and return CSV result"""
        sql_no_quotes = sql.replace('"', '')
        result = self.session.query(sql_no_quotes, 'CSV')
        return result.bytes().decode('utf-8').strip().replace('"', '').replace('\\N', 'NULL')

    def test_isnull_email_execution(self):
        """Test finding records with NULL email"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id", "name").filter(ds.email.isnull()).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Bob, David, Frank have NULL emails
        self.assertEqual(['2,Bob', '4,David', '6,Frank'], lines)

    def test_notnull_email_execution(self):
        """Test finding records with non-NULL email"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id").filter(ds.email.notnull()).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Alice, Charlie, Eve have emails
        self.assertEqual(['1', '3', '5'], lines)

    def test_null_or_null_execution(self):
        """Test OR of multiple NULL checks"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id").filter(ds.email.isnull() | ds.phone.isnull()).sort("id").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # Bob (NULL email), Charlie (NULL phone), David (both NULL), Frank (NULL email)
        self.assertEqual(['2', '3', '4', '6'], lines)

    def test_all_fields_null_execution(self):
        """Test finding records where all optional fields are NULL"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id", "name").filter(ds.email.isnull() & ds.phone.isnull() & ds.address.isnull()).to_sql()

        result = self._execute(sql)
        # Only David has all three fields NULL
        self.assertEqual('4,David', result)

    def test_null_and_status_execution(self):
        """Test NULL check combined with status filter"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id", "name").filter(ds.email.notnull() & (ds.status == 'active')).sort("id").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # Alice, Eve have email and are active
        self.assertEqual(['1,Alice', '5,Eve'], lines)

    def test_not_null_negation_execution(self):
        """Test NOT (IS NULL) execution"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id").filter(~ds.email.isnull()).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Same as notnull: 1, 3, 5
        self.assertEqual(['1', '3', '5'], lines)


if __name__ == '__main__':
    unittest.main()
