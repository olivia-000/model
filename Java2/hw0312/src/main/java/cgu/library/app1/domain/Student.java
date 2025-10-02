package cgu.library.app1.domain;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class Student {
    
    @Id
    @GeneratedValue(strategy = GenerationType.AUTO) // 自動產生主鍵
    private Long id; // JPA 需要一個主鍵，但不設 studentNo 為主鍵
    
    private String studentNo;
    private String telephone;
    private int birthMonth;
    private int birthday;

    public Student() {}

    public Student(String studentNo, String telephone, int birthMonth, int birthday) {
        this.studentNo = studentNo;
        this.telephone = telephone;
        this.birthMonth = birthMonth;
        this.birthday = birthday;
    }

    // Getter & Setter
    public String getStudentNo() { return studentNo; }
    public void setStudentNo(String studentNo) { this.studentNo = studentNo; }

    public String getTelephone() { return telephone; }
    public void setTelephone(String telephone) { this.telephone = telephone; }

    public int getBirthMonth() { return birthMonth; }
    public void setBirthMonth(int birthMonth) { this.birthMonth = birthMonth; }

    public int getBirthday() { return birthday; }
    public void setBirthday(int birthday) { this.birthday = birthday; }

    
}
