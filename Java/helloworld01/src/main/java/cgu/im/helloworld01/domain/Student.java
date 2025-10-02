package cgu.im.helloworld01.domain;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class Student {
	
	@Id
	@GeneratedValue(strategy = GenerationType.AUTO)
	private Long id;
	
	
	private String studentNo,telephone,dept;
	private int birthMonth,birthday;

	public Student(String studentNo, String telephone, int birthMonth, int birthday, String dept) {
		super();
		this.studentNo = studentNo;
		this.studentNo = telephone;
		this.studentNo = dept;
		this.birthMonth = birthMonth;
		this.birthday = birthday;
	}
	
	
	
	public String getStudentNo() {
		return studentNo;
	}
	
	public int getBirthMonth() {
		return birthMonth;
	}
	
	public int getBirthday() {
		return birthday;
	}

	public void setStudentNo(String studentNo) {
		this.studentNo = studentNo;
	}
	
	public void setTelephone(String telephone) {
		this.telephone = telephone;
	}
	
	public void setDept(String dept) {
		this.dept = dept;
	}
	
	public void setBirthMonth(int birthMonth) {
		this.birthMonth = birthMonth;
	}
	
	public void setBirthday(int birthday) {
		this.birthday = birthday;
	}

	public Long getId() {
		return id;
	}
	
	

}
