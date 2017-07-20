Assignemnt 2 for big data class in CUHK master program. This is an implementation of LU decomposition and recomposition. The matrix is provided with missing value. It is filled in with random value.

Requirement: 
	JDK 1.8+
	Maven
	
Compiling:
	Go to root folder where pom.xml resides
	> mvn clean install
	> cd target
	> java -jar Assignment2.CFMovie-1.0-SNAPSHOT.jar --train ..\train.dat --test ..\test.dat
	The prediction will be in the target folder
