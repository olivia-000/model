package cgu.library.app1;

import cgu.library.app1.domain.Student;
import cgu.library.app1.domain.StudentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.ArrayList;
import java.util.List;

@SpringBootApplication
public class App1Application implements CommandLineRunner {

    @Autowired
    private StudentRepository Repository;

    public static void main(String[] args) {
        SpringApplication.run(App1Application.class, args);
    }

    @Override
    public void run(String... args) {
        // 插入預設的 3 筆資料
        Repository.save(new Student("B1344000", "0912345678", 3, 13));
        Repository.save(new Student("B1345000", "0911111111", 1, 1));
        Repository.save(new Student("B1346000", "0912121212", 10, 1));

        // 1. 顯示所有學生資料，格式化輸出
        System.out.println("Student ID    Telephone   BMonth BDay");
        System.out.println("------------------------------------------------");

        Iterable<Student> studentIterable = Repository.findAll();
        List<Student> students = new ArrayList<>();
        studentIterable.forEach(students::add); // 轉換為 List<Student>

        for (Student s : students) {
            System.out.printf("%-12s %-11s %-6d %-6d%n", 
                s.getStudentNo(), s.getTelephone(), s.getBirthMonth(), s.getBirthday());
        }
        
        /*
        %-12s	左對齊 (-號) 的 字串 (s)，佔 12 個字元寬度
		%-11s	左對齊 的 字串 (s)，佔 11 個字元寬度
		%-6d	左對齊 的 整數 (d)，佔 6 個字元寬度
		%-6d	左對齊 的 整數 (d)，佔 6 個字元寬度
		%n		換行，確保跨平台換行格式正確
		*/

        // 2. 搜尋上半年出生的學生
        System.out.println("\n上半年出生的學生:");
        Repository.findByBirthMonthBetween(1, 6)
                         .forEach(student -> System.out.println(student.getStudentNo()));
    }
}
