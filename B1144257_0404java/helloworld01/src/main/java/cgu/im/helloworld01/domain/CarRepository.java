package cgu.im.helloworld01.domain;

import java.util.List;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

public interface CarRepository extends CrudRepository<Car, Long> {
	
	String q1 = "SELECT c FROM Car c ";
    String q2 = "WHERE c.registrationNumber LIKE '___-1%' "+ "OR ";
    String q3 = "c.registrationNumber LIKE '___-2%' "+ "OR ";
    String q4 = "c.registrationNumber LIKE '___-3%'";
    
	// Fetch cars by brand
     List<Car> findByBrand(String brand);
     
	 // Fetch cars by brand
     List<Car> findByColor(String color);
     
     // 
     List<Car> findAllByColor(String color);
     
     // Fetch cars by year
     List<Car> findByModelYear(int year);
     
     // Fetch cars by year
     List<Car> findByModelYearGreaterThanEqual(int year);
     
     // Fetch cars by brand and model
     List<Car> findByBrandAndModel(String brand, String model);
     
     // Fetch cars by brand or color
     List<Car> findByBrandOrColor(String brand, String color);
     
     // Fetch all cars and sort them by model_year
     List<Car> findByModelYearOrderByModelYear(int year); 
     
     // Fetch all cars and sort them by model_year
     List<Car> findOrderByModelYear(int year); 
     
     // Fetch all cars and sort them by model_year
     //List<Car> findByOrderByModelYear(int year); 
     
     // Fetch cars by year and print fetched cars by brand
     List<Car> findByModelYearOrderByBrand(int year);
     
     // Fetch cars based on brand and sort them by model_year descending
     List<Car> findByBrandOrderByModelYearDesc(String brand);
     
     // Delete a car c
     //void delete(Car c);
     
     // 請取得2022年以外年份出廠的所有汽車資料
     List<Car> findByModelYearNotOrderByModelYear(int year);
     
     //
     @Query("SELECT c FROM Car c WHERE c.registrationNumber LIKE '____-__'")
     List<Car> findCarByRegistrationNumberFifthLetter();
     
     //JPQL查詢車廠為Toyota者
     @Query("SELECT c FROM Car c WHERE c.brand LIKE 'Toyota'")
     List<Car> findToyota();
     
     //JPQL取得所有車子的平均價格
     @Query("SELECT AVG(c.price) FROM Car c")
     double findAvgCarPrice(); 
     
     // 取得汽車最低價
     @Query("SELECT MIN(c.price) FROM Car c")
     double findMinCarPrice(); 
     
     // JPQL取得所有車子的平均價格    in CarRepository interface
     @Query("SELECT COUNT(c) FROM Car c")
     int findCarNumber();
     
     // JPQL取得所有車子的平均價格    in CarRepository interface
     @Query("SELECT COUNT(DISTINCT c.brand) FROM Car c")
     int findCarBrandNumber();
     
     // JPQL取得所有車價總和的12%
     @Query("SELECT SUM(c.price * 0.12) FROM Car c")
     double findPriceTax(); 
     
     //JPQL查詢紅色車輛
     @Query("SELECT c FROM Car c WHERE c.color = 'Red'")
     List<Car> findRedCars();
     
     //JPQL查詢車輛廠牌是舊式車牌者(第五字母是'-')
     @Query("SELECT c FROM Car c WHERE c.registrationNumber LIKE '____-__'")
     List<Car> findCarByRegistrationNumberFifthLetter1();
     
     // JPQL查詢車號包含02者
     @Query("SELECT c FROM Car c WHERE c.registrationNumber LIKE '%02%'")
     List<Car> findCarByRegistrationNumberContaining02();
     
     // JPQL查詢車號包含兩個02者
     @Query("SELECT c FROM Car c WHERE c.registrationNumber LIKE '%2%2%'")
     List<Car> findCarByRegistrationNumberContaining2AtLeastTwice();
     
     @Query(value=q1+q2+q3+q4)
     List<Car> findCarByNewRegistrationNumberSecondPartBeginWith1Or2Or3();
     
     










}
