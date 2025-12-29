"""
第1周：Python基础示例（重命名自 beginner_ai）
"""

__version__ = "1.0.0"
__author__ = "AI Learning Team"

class Student:
    """学生类"""
    def __init__(self, student_id: str, name: str, grade: int):
        self.student_id = student_id
        self.name = name
        self.grade = grade
        self.scores = {}

    def add_score(self, subject: str, score: float):
        if 0 <= score <= 100:
            self.scores[subject] = score
            return True
        return False

    def get_average(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def __str__(self):
        avg = self.get_average()
        return f"学号:{self.student_id} 姓名:{self.name} 年级:{self.grade} 平均分:{avg:.2f}"


class GradeManager:
    """成绩管理系统"""
    def __init__(self):
        self.students = {}

    def add_student(self, student: Student):
        self.students[student.student_id] = student
        print(f"✓ 添加学生: {student.name}")

    def remove_student(self, student_id: str):
        if student_id in self.students:
            name = self.students[student_id].name
            del self.students[student_id]
            print(f"✓ 删除学生: {name}")
            return True
        return False

    def find_student(self, student_id: str) -> Student:
        return self.students.get(student_id)

    def get_top_students(self, n: int = 3):
        sorted_students = sorted(
            self.students.values(),
            key=lambda s: s.get_average(),
            reverse=True
        )
        return sorted_students[:n]

    def get_statistics(self):
        if not self.students:
            return None
        all_averages = [s.get_average() for s in self.students.values()]
        return {
            "total_students": len(self.students),
            "class_average": sum(all_averages) / len(all_averages),
            "highest_average": max(all_averages),
            "lowest_average": min(all_averages)
        }

    def display_all(self):
        if not self.students:
            print("暂无学生记录")
            return
        print("\n" + "="*60)
        print("学生成绩一览表")
        print("="*60)
        for student in self.students.values():
            print(student)
            if student.scores:
                for subject, score in student.scores.items():
                    print(f"  - {subject}: {score}分")
        print("="*60)


def demonstrate_python_basics():
    print("\n" + "="*60)
    print("🐍 第1周：Python基础语法与面向对象编程")
    print("="*60)

    print("\n1. 基本语法示例")
    squares = [x**2 for x in range(5)]
    square_map = {x: x**2 for x in range(5)}
    doubled = [x*2 for x in [1, 2, 3, 4]]

    print("  • 列表推导式:", squares)
    print("  • 字典推导式:", square_map)
    print("  • Lambda函数:", doubled)

    print("\n2. 函数式编程示例")
    def decorator_example(func):
        def wrapper(*args, **kwargs):
            print(f"  调用函数: {func.__name__}")
            result = func(*args, **kwargs)
            print(f"  返回结果: {result}")
            return result
        return wrapper

    @decorator_example
    def add(a, b):
        return a + b

    add(3, 5)

    print("\n3. 面向对象编程 - 学生成绩管理系统")
    manager = GradeManager()

    s1 = Student("2024001", "张三", 10)
    s1.add_score("数学", 95)
    s1.add_score("语文", 88)
    s1.add_score("英语", 92)
    manager.add_student(s1)

    s2 = Student("2024002", "李四", 10)
    s2.add_score("数学", 87)
    s2.add_score("语文", 90)
    s2.add_score("英语", 85)
    manager.add_student(s2)

    s3 = Student("2024003", "王五", 10)
    s3.add_score("数学", 92)
    s3.add_score("语文", 95)
    s3.add_score("英语", 89)
    manager.add_student(s3)

    manager.display_all()

    print("\n4. 统计信息")
    stats = manager.get_statistics()
    print(f"  总人数: {stats['total_students']}")
    print(f"  班级平均分: {stats['class_average']:.2f}")
    print(f"  最高平均分: {stats['highest_average']:.2f}")
    print(f"  最低平均分: {stats['lowest_average']:.2f}")

    print("\n5. 前三名")
    top3 = manager.get_top_students(3)
    for i, student in enumerate(top3, 1):
        print(f"  第{i}名: {student.name} - {student.get_average():.2f}分")

    print("\n6. 异常处理示例")
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        print(f"  ✓ 捕获异常: {e}")
    finally:
        print("  ✓ 清理资源完成")

    print("\n7. 文件操作示例")
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        temp_path = f.name
        f.write("Hello, Python!\n")
        f.write("这是文件操作示例\n")
        print(f"  ✓ 写入临时文件: {temp_path}")

    with open(temp_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"  ✓ 读取内容: {content.strip()}")

    os.unlink(temp_path)
    print("  ✓ 删除临时文件")

    print("\n" + "="*60)
    print("✅ 第1周学习完成!")
    print("="*60)
    return manager


if __name__ == "__main__":
    demonstrate_python_basics()
