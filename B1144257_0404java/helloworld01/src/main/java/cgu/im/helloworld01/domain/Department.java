package cgu.im.helloworld01.domain;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class Department {
	
	@Id
	@GeneratedValue(strategy = GenerationType.AUTO)
	private Long id;
	
	private String deptNo,deptTitle;
	private int deptFloor;
	
	public Department() {}
	
	public Department(String deptNo, String deptTitle, int deptFloor) {
		super();
		this.deptNo = deptNo;
		this.deptTitle = deptTitle;
		this.deptFloor = deptFloor;
	}
	
	
	
	// 補齊所有 getter 方法（Console 顯示資料需要）

	public void setDeptTitle(String deptTitle) {
		this.deptTitle = deptTitle;
	}
	
	public void setDeptNo(String deptNo) {
		this.deptNo = deptNo;
	}
	
	public void setDeptFloor(int deptFloor) {
		this.deptFloor = deptFloor;
	}
	
	public Long getId() {
		return id;
	}
	
	public String getDeptNo() {
		return deptNo;
	}
	
	public String getDeptTitle() {
		return deptTitle;
	}
	
	public int getDeptFloor() {
		return deptFloor;
	}

	



}
