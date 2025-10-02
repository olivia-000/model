package cgu.library.app1.domain;

import org.springframework.data.repository.CrudRepository;
import java.util.List;

public interface StudentRepository extends CrudRepository<Student, Long> {
    // 自定義查詢方法：搜尋上半年（1~6月）出生的學生
    List<Student> findByBirthMonthBetween(int start, int end);
}
