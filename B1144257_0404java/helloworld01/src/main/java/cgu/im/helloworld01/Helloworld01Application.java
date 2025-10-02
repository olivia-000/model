package cgu.im.helloworld01;

import java.lang.reflect.Field;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import cgu.im.helloworld01.domain.Car;
import cgu.im.helloworld01.domain.CarRepository;
import cgu.im.helloworld01.domain.Department;
import cgu.im.helloworld01.domain.DepartmentRepository;
import cgu.im.helloworld01.domain.Owner;
import cgu.im.helloworld01.domain.OwnerRepository;
import cgu.im.helloworld01.domain.Student;
import cgu.im.helloworld01.domain.StudentRepository;

@SpringBootApplication
public class Helloworld01Application implements CommandLineRunner {
	
	private static final Logger logger = 			
			LoggerFactory.getLogger(Helloworld01Application.class);
	
	private final CarRepository repository;
	private final StudentRepository Repository;
	private final OwnerRepository orepository;
	private final DepartmentRepository Drepository;
	
	public Helloworld01Application(CarRepository repository, OwnerRepository orepository, StudentRepository Repository, DepartmentRepository Drepository) {
	       this.repository = repository;
	       this.orepository = orepository;
	       this.Repository = Repository;
	       this.Drepository = Drepository;
	}

	public static void main(String[] args) {
		
		SpringApplication.run(Helloworld01Application.class, args);
		logger.info("Application started");
	}

	@Override
	public void run(String... args) throws Exception {
		// TODO Auto-generated method stub	
		
		Owner owner1 = new Owner("John" , "Johnson");
		Owner owner2 = new Owner("Mary" , "Robinson");
		orepository.save(owner1);
		orepository.save(owner2);


		repository.save(new Car("Ford", "Mustang", "Red", "ADF-1121", 2023, 59000, owner1));		
		repository.save(new Car("Nissan", "Leaf", "White", "SSJ-3002", 2020, 29000, owner1));
		repository.save(new Car("Toyota", "Prius", "Silver", "KKO-0212", 2022, 39000, owner1));
		repository.save(new Car("Nissan", "Tiida", "White", "SSJ-3003", 2021, 17000, owner2));
		repository.save(new Car("Toyota", "Altis", "Silver", "KKO-0213", 2024, 29000, owner2));
		repository.save(new Car("Toyota", "Mustang", "Red","0212-DD", 2023, 59000, owner2));
		
		Repository.save(new Student("B1144257", "0912345678",3,13,"D001"));
		Repository.save(new Student("B1345000", "0911111111",1,1,"D001"));
		Repository.save(new Student("B1346000", "0912121212",10,1,"D002"));
		
		Drepository.save(new Department("D001", "資訊管理 ", 5));
		Drepository.save(new Department("D002", "工商管理 ", 7));
		Drepository.save(new Department("D003", "醫務管理 ", 6));
		
		
		System.out.println("JPQL取得所有車子的平均價格: "+repository.findAvgCarPrice());
		
		System.out.println("JPQL找出最便宜的車價: "+repository.findMinCarPrice());
		
		System.out.println("JPQL求得所有汽車的數量: "+repository.findCarNumber());
		
		System.out.println("JPQL計算不同車廠的數量: "+repository.findCarBrandNumber());
		
		// In run() of HelloworldApplication class
	    System.out.println("JPQL計算所有車價總和的12%  : "+repository.findPriceTax());
	    
	    //針對特定欄位進行篩選
	    System.out.println("JPQL查詢紅色車輛的汽車廠牌、車號、出廠年份: ");
	    for (Car car : repository.findRedCars()) {	    	 
	           logger.info("brand: {}, car#: {}, year: {}, color: {}",
	                       car.getBrand(), car.getRegistrationNumber(), car.getModelYear(), car.getColor());
	    }
	    
	    //查詢車輛廠牌是舊式車牌者
	    System.out.println("JPQL查詢車輛廠牌是舊式車牌者 ");
	    for (Car car : repository.findCarByRegistrationNumberFifthLetter()) {	    	 
	           logger.info("brand: {}, car#: {}, year: {}",
	                    car.getBrand(), car.getRegistrationNumber(), car.getModelYear());
	    }
	    
	    //查詢車號包含02者
	    System.out.println("JPQL查詢車號包含02者 ");	     
	    for (Car car : repository.findCarByRegistrationNumberContaining02()) {	 
	              logger.info("brand: {}, car#: {}, year: {}",
	                    car.getBrand(), car.getRegistrationNumber(), car.getModelYear());
	    }
	    
	    //查詢車號包含兩個02者
	    System.out.println("JPQL查詢車號至少包含兩個2者 ");
	    for (Car car : repository.findCarByRegistrationNumberContaining2AtLeastTwice()) {	     
	                logger.info("brand: {}, car#: {}, year: {}",
	                    car.getBrand(), car.getRegistrationNumber(), car.getModelYear());
	    }
	    
	    //查詢新式車號後四碼為1,2,3開頭者 
	    System.out.println("JPQL查詢新式車號後四碼為1,2,3開頭者 ");
	    for (Car car : repository.findCarByNewRegistrationNumberSecondPartBeginWith1Or2Or3()) {	     
	           logger.info("brand: {}, car#: {}, year: {}",
	                    car.getBrand(), car.getRegistrationNumber(), car.getModelYear());
	    }
	    
	    // 取得所有車主的所有汽車資料     // 先印出車主編號、姓氏   //   然後逐行印出他名下汽車的車號
	    for (Owner owner : orepository.findAllWithCars()) {	    	       
	          System.out.printf("owner#: %s last_name: %s\n", 
	          owner.getOwnerid(), owner.getLastname());	            	            
	          List<Car> cars = owner.getCars();
	    	            
	          if(cars == null) {
	    	System.out.println(owner.getLastname() + "has no car"); 
	          }
	          else {	            	
	            	for(Car cc: cars) {
	    	      System.out.println(cc.getRegistrationNumber());
	    	}		            
	           }
	    }
	    
	    Iterable<Department> departments = Drepository.findAll();
        
        // 輸出表格標題
        System.out.println("系所代號\t系所名稱\t所在樓層");
        System.out.println("================================");
        
        // 遍歷資料並格式化輸出
        departments.forEach(dept -> 
            System.out.printf("%-8s\t%-10s\t%d樓%n",
                dept.getDeptNo(),
                dept.getDeptTitle(),
                dept.getDeptFloor())
        		 );
	    
	    
	    
	    
	    

	    















			
	}

}
