"""
Tests for utility functions in ragu/utils/ragu_utils.py.
"""
import asyncio
from unittest.mock import Mock

import pytest
from aiolimiter import AsyncLimiter

from ragu.utils.ragu_utils import (
    compute_mdhash_id,
    always_get_an_event_loop,
    AsyncRunner
)


class TestComputeMdhashId:
    """
    Tests for compute_mdhash_id function.
    """

    def test_basic_hash_generation(self):
        """
        Test basic hash generation without prefix.
        """
        content = "test content"
        hash_id = compute_mdhash_id(content)

        assert isinstance(hash_id, str)
        assert len(hash_id) == 32  # MD5 hash length
        # Verify it's hexadecimal
        assert all(c in "0123456789abcdef" for c in hash_id)

    def test_hash_with_prefix(self):
        """
        Test hash generation with prefix.
        """
        content = "test content"
        prefix = "entity-"
        hash_id = compute_mdhash_id(content, prefix=prefix)

        assert hash_id.startswith(prefix)
        assert len(hash_id) == len(prefix) + 32

    def test_deterministic_hashing(self):
        """Test that same content produces same hash."""
        content = "test content"
        hash1 = compute_mdhash_id(content)
        hash2 = compute_mdhash_id(content)

        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Test that different content produces different hashes."""
        hash1 = compute_mdhash_id("content1")
        hash2 = compute_mdhash_id("content2")

        assert hash1 != hash2

    def test_empty_string(self):
        hash_id = compute_mdhash_id("")

        assert isinstance(hash_id, str)
        assert len(hash_id) == 32

    def test_unicode_content(self):
        content = "Hello 世界 мир"
        hash_id = compute_mdhash_id(content)

        assert isinstance(hash_id, str)
        assert len(hash_id) == 32

    def test_special_characters(self):
        content = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        hash_id = compute_mdhash_id(content)

        assert isinstance(hash_id, str)
        assert len(hash_id) == 32

    def test_long_content(self):
        content = "a" * 10000
        hash_id = compute_mdhash_id(content)

        assert isinstance(hash_id, str)
        assert len(hash_id) == 32  # MD5 always produces same length


class TestAlwaysGetAnEventLoop:
    def test_gets_existing_loop(self):
        """Test that it returns existing event loop when available."""
        # Create a new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = always_get_an_event_loop()

        assert result is loop
        assert not result.is_closed()

        # Cleanup
        loop.close()

    def test_creates_new_loop_when_closed(self):
        """Test that it creates new loop when current is closed."""
        # Create and close a loop
        old_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(old_loop)
        old_loop.close()

        result = always_get_an_event_loop()

        assert result is not old_loop
        assert not result.is_closed()

        # Cleanup
        result.close()

    def test_creates_loop_when_none_exists(self):
        """Test that it creates loop when none exists."""
        # Try to clear event loop
        try:
            asyncio.set_event_loop(None)
        except RuntimeError:
            pass

        result = always_get_an_event_loop()

        assert result is not None
        assert isinstance(result, asyncio.AbstractEventLoop)
        assert not result.is_closed()

        # Cleanup
        result.close()

    def test_loop_is_usable(self):
        """Test that returned loop can actually run tasks."""
        loop = always_get_an_event_loop()

        async def simple_task():
            return "success"

        result = loop.run_until_complete(simple_task())
        assert result == "success"

        # Cleanup
        loop.close()


class TestAsyncRunner:
    @pytest.fixture
    def mock_progress_bar(self):
        pbar = Mock()
        pbar.update = Mock()
        return pbar

    @pytest.fixture
    def async_runner(self, mock_progress_bar):
        semaphore = asyncio.Semaphore(2)
        rps_limiter = AsyncLimiter(5, 1)  # 5 requests per second
        rpm_limiter = AsyncLimiter(60, 60)  # 60 requests per minute

        return AsyncRunner(semaphore, rps_limiter, rpm_limiter, mock_progress_bar)

    @pytest.mark.asyncio
    async def test_make_request_success(self, async_runner, mock_progress_bar):
        async def test_func(value):
            return value * 2

        result = await async_runner.make_request(test_func, value=5)

        assert result == 10
        mock_progress_bar.update.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_make_request_updates_progress_bar(self, async_runner, mock_progress_bar):
        async def simple_func():
            return "done"

        await async_runner.make_request(simple_func)

        mock_progress_bar.update.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_make_request_with_exception(self, async_runner, mock_progress_bar):
        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await async_runner.make_request(failing_func)

        # Progress bar should still be updated
        mock_progress_bar.update.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, mock_progress_bar):
        semaphore = asyncio.Semaphore(1)  # Only 1 concurrent
        rps_limiter = AsyncLimiter(100, 1)
        rpm_limiter = AsyncLimiter(1000, 60)

        runner = AsyncRunner(semaphore, rps_limiter, rpm_limiter, mock_progress_bar)

        execution_order = []

        async def tracked_func(task_id):
            execution_order.append(f"start-{task_id}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end-{task_id}")
            return task_id

        # Run two tasks concurrently
        tasks = [
            runner.make_request(tracked_func, task_id=1),
            runner.make_request(tracked_func, task_id=2)
        ]

        await asyncio.gather(*tasks)

        # With semaphore=1, first task should complete before second starts
        assert execution_order.index("end-1") < execution_order.index("start-2")

    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_progress_bar):
        semaphore = asyncio.Semaphore(10)
        rps_limiter = AsyncLimiter(2, 1)  # 2 requests per second
        rpm_limiter = AsyncLimiter(100, 60)

        runner = AsyncRunner(semaphore, rps_limiter, rpm_limiter, mock_progress_bar)

        import time
        start_time = time.time()

        async def quick_func(n):
            return n

        # Make 5 requests (should take at least 2 seconds with 2 req/sec limit)
        tasks = [runner.make_request(quick_func, n=i) for i in range(5)]
        await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # Should take at least 2 seconds (5 requests / 2 per second = 2.5 seconds)
        # Using 1.5 to account for timing variance
        assert elapsed >= 1.5

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, async_runner, mock_progress_bar):
        async def async_func(value):
            await asyncio.sleep(0.001)
            return value ** 2

        tasks = [async_runner.make_request(async_func, value=i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert results == [i ** 2 for i in range(10)]
        assert mock_progress_bar.update.call_count == 10
