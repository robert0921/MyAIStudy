"""
Week 18: 系统监控与异常恢复
包括：日志分析、性能监控、异常检测、自动恢复

本模块演示生产级系统的监控和运维（模拟Prometheus/ELK等工具）
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
import time
import random
import json


class Metric:
    """指标对象"""
    
    def __init__(self, name: str, value: float, labels: Optional[Dict] = None, 
                 timestamp: Optional[datetime] = None):
        self.name = name
        self.value = value
        self.labels = labels or {}
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'labels': self.labels,
            'timestamp': self.timestamp.isoformat()
        }


class PrometheusCollector:
    """Prometheus风格的指标收集器（模拟）"""
    
    def __init__(self):
        self.metrics = []
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
    
    def counter_inc(self, name: str, value: float = 1.0, labels: Optional[Dict] = None):
        """计数器自增"""
        key = self._make_key(name, labels)
        self.counters[key] += value
        self.metrics.append(Metric(name, self.counters[key], labels))
    
    def gauge_set(self, name: str, value: float, labels: Optional[Dict] = None):
        """设置仪表值"""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        self.metrics.append(Metric(name, value, labels))
    
    def histogram_observe(self, name: str, value: float, labels: Optional[Dict] = None):
        """直方图观察"""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
        self.metrics.append(Metric(name, value, labels))
    
    def _make_key(self, name: str, labels: Optional[Dict]) -> str:
        """生成指标键"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def get_metrics(self, name: Optional[str] = None) -> List[Metric]:
        """获取指标"""
        if name:
            return [m for m in self.metrics if m.name == name]
        return self.metrics
    
    def get_counter(self, name: str, labels: Optional[Dict] = None) -> float:
        """获取计数器值"""
        key = self._make_key(name, labels)
        return self.counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict] = None) -> float:
        """获取仪表值"""
        key = self._make_key(name, labels)
        return self.gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict] = None) -> Dict:
        """获取直方图统计"""
        key = self._make_key(name, labels)
        values = self.histograms.get(key, [])
        
        if not values:
            return {}
        
        import numpy as np
        return {
            'count': len(values),
            'sum': sum(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }


class LogLevel:
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntry:
    """日志条目"""
    
    def __init__(self, level: str, message: str, extra: Optional[Dict] = None):
        self.level = level
        self.message = message
        self.extra = extra or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'message': self.message,
            **self.extra
        }
    
    def to_json(self) -> str:
        """ELK风格的JSON日志"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class Logger:
    """结构化日志器（ELK风格）"""
    
    def __init__(self, name: str):
        self.name = name
        self.logs = deque(maxlen=1000)
    
    def _log(self, level: str, message: str, **kwargs):
        """记录日志"""
        entry = LogEntry(level, message, {'logger': self.name, **kwargs})
        self.logs.append(entry)
        
        # 打印到控制台
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level:8s} {self.name}: {message}")
        
        if kwargs:
            print(f"  └─ {kwargs}")
    
    def debug(self, message: str, **kwargs):
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def get_logs(self, level: Optional[str] = None, last_n: Optional[int] = None) -> List[LogEntry]:
        """获取日志"""
        logs = list(self.logs)
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        if last_n:
            logs = logs[-last_n:]
        
        return logs


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 20, threshold_std: float = 2.0):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.history = deque(maxlen=window_size)
    
    def add_value(self, value: float):
        """添加值"""
        self.history.append(value)
    
    def is_anomaly(self, value: float) -> Tuple[bool, float]:
        """检测是否为异常"""
        if len(self.history) < self.window_size:
            return False, 0.0
        
        import numpy as np
        mean = np.mean(self.history)
        std = np.std(self.history)
        
        if std == 0:
            return False, 0.0
        
        # Z-score
        z_score = abs(value - mean) / std
        is_anomaly = z_score > self.threshold_std
        
        return is_anomaly, z_score


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = []
    
    def add_rule(self, name: str, condition: callable, severity: str):
        """添加告警规则"""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity
        })
    
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """检查告警"""
        triggered = []
        
        for rule in self.alert_rules:
            if rule['condition'](metrics):
                alert = {
                    'name': rule['name'],
                    'severity': rule['severity'],
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                }
                self.alerts.append(alert)
                triggered.append(alert)
        
        return triggered
    
    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """获取最近的告警"""
        return self.alerts[-n:]


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_fn: callable):
        """注册健康检查"""
        self.checks[name] = check_fn
    
    def run_checks(self) -> Dict:
        """运行所有检查"""
        results = {}
        all_healthy = True
        
        for name, check_fn in self.checks.items():
            try:
                is_healthy = check_fn()
                results[name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'healthy': is_healthy
                }
                if not is_healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'healthy': False,
                    'error': str(e)
                }
                all_healthy = False
        
        return {
            'overall': 'healthy' if all_healthy else 'unhealthy',
            'checks': results,
            'timestamp': datetime.now().isoformat()
        }


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: callable, *args, **kwargs):
        """通过熔断器调用函数"""
        # 检查状态
        if self.state == "OPEN":
            # 检查是否可以尝试恢复
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).seconds > self.timeout_seconds:
                self.state = "HALF_OPEN"
                print(f"🔄 熔断器进入半开状态，尝试恢复...")
            else:
                raise Exception(f"熔断器开启，拒绝请求")
        
        try:
            result = func(*args, **kwargs)
            
            # 成功，重置计数
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                print(f"✅ 熔断器关闭，服务恢复正常")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            # 检查是否需要开启熔断
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                print(f"🚨 熔断器开启！连续失败 {self.failure_count} 次")
            
            raise e


def demonstrate_metrics_collection():
    """演示指标收集"""
    print("\n" + "="*70)
    print("📊 演示：Prometheus指标收集")
    print("="*70)
    
    collector = PrometheusCollector()
    
    print("\n【模拟API请求】")
    
    # 模拟一系列请求
    for i in range(20):
        # 请求计数
        collector.counter_inc("http_requests_total", labels={'method': 'GET', 'path': '/api/chat'})
        
        # 随机延迟
        latency = random.uniform(0.05, 0.5)
        collector.histogram_observe("http_request_duration_seconds", latency, 
                                    labels={'method': 'GET'})
        
        # CPU使用率
        cpu_usage = random.uniform(20, 80)
        collector.gauge_set("cpu_usage_percent", cpu_usage)
        
        # 内存使用
        memory_mb = random.uniform(500, 2000)
        collector.gauge_set("memory_usage_mb", memory_mb)
        
        # 随机产生一些错误
        if random.random() < 0.1:
            collector.counter_inc("http_requests_total", labels={'method': 'GET', 'status': '500'})
    
    # 显示指标
    print(f"\n【收集的指标】")
    print(f"总请求数: {collector.get_counter('http_requests_total', {'method': 'GET', 'path': '/api/chat'})}")
    print(f"当前CPU使用率: {collector.get_gauge('cpu_usage_percent'):.2f}%")
    print(f"当前内存使用: {collector.get_gauge('memory_usage_mb'):.2f} MB")
    
    # 延迟统计
    latency_stats = collector.get_histogram_stats("http_request_duration_seconds", {'method': 'GET'})
    print(f"\n【请求延迟统计】")
    for key, value in latency_stats.items():
        if isinstance(value, float):
            print(f"  {key:6s}: {value:.3f}s")
        else:
            print(f"  {key:6s}: {value}")


def demonstrate_structured_logging():
    """演示结构化日志"""
    print("\n" + "="*70)
    print("📝 演示：结构化日志（ELK风格）")
    print("="*70)
    
    logger = Logger("api-service")
    
    print("\n【不同级别的日志】")
    
    logger.info("服务启动", port=8000, version="1.0.0")
    logger.debug("加载配置文件", config_path="/etc/app/config.yaml")
    logger.info("数据库连接成功", db_host="localhost", db_name="myapp")
    logger.warning("请求处理缓慢", latency_ms=1500, threshold_ms=1000)
    logger.error("API调用失败", error="Connection timeout", retry_count=3)
    logger.critical("数据库连接丢失", db_host="localhost", action="尝试重连")
    
    # 统计日志
    print(f"\n【日志统计】")
    print(f"总日志数: {len(logger.logs)}")
    print(f"错误日志数: {len(logger.get_logs(LogLevel.ERROR))}")
    print(f"警告日志数: {len(logger.get_logs(LogLevel.WARNING))}")
    
    # 导出JSON格式
    print(f"\n【JSON格式日志示例】")
    recent_logs = logger.get_logs(last_n=2)
    for log in recent_logs:
        print(log.to_json())


def demonstrate_anomaly_detection():
    """演示异常检测"""
    print("\n" + "="*70)
    print("🔍 演示：异常检测")
    print("="*70)
    
    detector = AnomalyDetector(window_size=20, threshold_std=2.0)
    
    print("\n【模拟指标数据流】")
    print(f"窗口大小: {detector.window_size}")
    print(f"异常阈值: {detector.threshold_std} 标准差")
    
    # 正常数据
    print("\n正常数据 (延迟在100-200ms之间):")
    for i in range(20):
        value = random.uniform(100, 200)
        detector.add_value(value)
        print(f"  样本 {i+1}: {value:.1f}ms")
    
    # 测试异常值
    print("\n【异常检测测试】")
    test_values = [150, 180, 350, 120, 500, 160]
    
    for value in test_values:
        is_anomaly, z_score = detector.is_anomaly(value)
        detector.add_value(value)
        
        status = "🚨 异常" if is_anomaly else "✅ 正常"
        print(f"  值: {value:6.1f}ms, Z-score: {z_score:.2f}, 状态: {status}")


def demonstrate_alerting():
    """演示告警系统"""
    print("\n" + "="*70)
    print("🚨 演示：告警管理")
    print("="*70)
    
    alert_manager = AlertManager()
    
    # 定义告警规则
    alert_manager.add_rule(
        name="高CPU使用率",
        condition=lambda m: m.get('cpu_usage', 0) > 80,
        severity="WARNING"
    )
    
    alert_manager.add_rule(
        name="错误率过高",
        condition=lambda m: m.get('error_rate', 0) > 0.05,
        severity="CRITICAL"
    )
    
    alert_manager.add_rule(
        name="响应时间过长",
        condition=lambda m: m.get('latency_ms', 0) > 1000,
        severity="WARNING"
    )
    
    print(f"\n【告警规则】")
    for rule in alert_manager.alert_rules:
        print(f"  - {rule['name']} (级别: {rule['severity']})")
    
    # 模拟指标监控
    print(f"\n【指标监控】")
    
    test_cases = [
        {'cpu_usage': 75, 'error_rate': 0.01, 'latency_ms': 200, 'desc': '正常'},
        {'cpu_usage': 85, 'error_rate': 0.02, 'latency_ms': 300, 'desc': 'CPU高'},
        {'cpu_usage': 60, 'error_rate': 0.08, 'latency_ms': 400, 'desc': '错误率高'},
        {'cpu_usage': 70, 'error_rate': 0.03, 'latency_ms': 1200, 'desc': '响应慢'},
    ]
    
    for i, metrics in enumerate(test_cases, 1):
        print(f"\n时刻 {i} ({metrics['desc']}):")
        print(f"  CPU: {metrics['cpu_usage']}%, 错误率: {metrics['error_rate']:.1%}, 延迟: {metrics['latency_ms']}ms")
        
        alerts = alert_manager.check_alerts(metrics)
        if alerts:
            for alert in alerts:
                print(f"  🚨 触发告警: {alert['name']} [{alert['severity']}]")
        else:
            print(f"  ✅ 无告警")
    
    # 显示历史告警
    print(f"\n【历史告警】")
    recent_alerts = alert_manager.get_recent_alerts(n=5)
    for alert in recent_alerts:
        print(f"  - {alert['timestamp'][:19]} | {alert['severity']:8s} | {alert['name']}")


def demonstrate_health_check():
    """演示健康检查"""
    print("\n" + "="*70)
    print("💚 演示：健康检查")
    print("="*70)
    
    health_checker = HealthChecker()
    
    # 注册检查项
    health_checker.register_check("database", lambda: random.random() > 0.2)
    health_checker.register_check("redis", lambda: random.random() > 0.1)
    health_checker.register_check("api_service", lambda: True)
    health_checker.register_check("disk_space", lambda: random.random() > 0.05)
    
    print(f"\n【健康检查】")
    
    for i in range(3):
        print(f"\n第 {i+1} 次检查:")
        result = health_checker.run_checks()
        
        print(f"总体状态: {result['overall']}")
        print(f"检查项:")
        
        for name, check_result in result['checks'].items():
            status_icon = "✅" if check_result['healthy'] else "❌"
            print(f"  {status_icon} {name:15s}: {check_result['status']}")
        
        if i < 2:
            time.sleep(1)


def demonstrate_circuit_breaker():
    """演示熔断器"""
    print("\n" + "="*70)
    print("🔌 演示：熔断器")
    print("="*70)
    
    breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=5)
    
    # 模拟一个不稳定的服务
    call_count = 0
    
    def unstable_service():
        nonlocal call_count
        call_count += 1
        
        # 前5次调用会失败
        if call_count <= 5:
            raise Exception(f"服务调用失败")
        
        return "服务调用成功"
    
    print(f"\n【测试熔断器】")
    print(f"失败阈值: {breaker.failure_threshold}")
    print(f"恢复超时: {breaker.timeout_seconds}秒")
    
    # 测试调用
    for i in range(10):
        print(f"\n调用 {i+1}:")
        print(f"  熔断器状态: {breaker.state}")
        
        try:
            result = breaker.call(unstable_service)
            print(f"  ✅ {result}")
        except Exception as e:
            print(f"  ❌ {str(e)}")
        
        # 在熔断器开启后等待
        if breaker.state == "OPEN" and i == 5:
            print(f"\n等待 {breaker.timeout_seconds} 秒让熔断器恢复...")
            time.sleep(breaker.timeout_seconds + 1)
        
        time.sleep(0.5)


def run_week18_demo():
    """运行Week 18完整演示"""
    print("\n" + "="*70)
    print("🚀 Week 18: 系统监控与异常恢复 - 完整演示")
    print("="*70)
    
    # 1. 指标收集
    demonstrate_metrics_collection()
    
    input("\n按Enter继续查看结构化日志...")
    
    # 2. 结构化日志
    demonstrate_structured_logging()
    
    input("\n按Enter继续查看异常检测...")
    
    # 3. 异常检测
    demonstrate_anomaly_detection()
    
    input("\n按Enter继续查看告警系统...")
    
    # 4. 告警系统
    demonstrate_alerting()
    
    input("\n按Enter继续查看健康检查...")
    
    # 5. 健康检查
    demonstrate_health_check()
    
    input("\n按Enter继续查看熔断器...")
    
    # 6. 熔断器
    demonstrate_circuit_breaker()
    
    print("\n" + "="*70)
    print("✅ Week 18演示完成！")
    print("="*70)
    print("\n核心收获：")
    print("  1. 掌握了Prometheus风格的指标收集")
    print("  2. 学会了ELK风格的结构化日志")
    print("  3. 实现了基于统计的异常检测")
    print("  4. 建立了完整的告警管理系统")
    print("  5. 理解了健康检查的重要性")
    print("  6. 掌握了熔断器的自动恢复机制")


if __name__ == "__main__":
    run_week18_demo()
