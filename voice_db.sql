create database voiceDB;
use voiceDB;

create table voice_data (
	`name` varchar(50),
    coefs text not null,
    
    primary key(`name`)
);

delete from voice_data;

select * from voice_data;

