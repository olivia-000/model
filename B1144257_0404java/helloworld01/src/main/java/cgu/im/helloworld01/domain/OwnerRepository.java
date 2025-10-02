package cgu.im.helloworld01.domain;

import java.util.List;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

public interface OwnerRepository extends CrudRepository<Owner, Long>
{
	// JPQL取得所有車主的所有汽車資料
    @Query("SELECT o FROM Owner o JOIN o.cars")
    List<Owner> findAllWithCars(); 
}
