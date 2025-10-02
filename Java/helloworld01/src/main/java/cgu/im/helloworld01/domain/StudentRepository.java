package cgu.im.helloworld01.domain;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.data.repository.CrudRepository;

public interface StudentRepository extends CrudRepository<Student, Long> {
	
	public static void printStudentsNotBornOnX(List<Student> students, int x) {
	    List<String> filteredStudents = students.stream()
	        // 過濾條件：生日"日期"不等於 X
	        .filter(s -> s.getBirthday() != x)  // 假設有 getBirthDay() 方法返回 int 型態的日期
	        // 排序：按生日"月份"升序
	        .sorted(Comparator.comparingInt(Student::getBirthMonth))
	        
	     // 使用 Lambda 表達式提取學號並收集為列表
	        List<Integer> studentIds = students.stream()
	            .map(student -> student.getStudentId()) // 使用 Lambda 表達式
	            .collect(Collectors.toList());

	    // 格式化輸出
	    System.out.println
	    ("符合條件的學號（按月份排序）：");
	    filteredStudents.forEach(System.out::println);
	    
	}
}
