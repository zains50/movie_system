create TABLE WEBSITE_USER ( 
    user_id UUID PRIMARY KEY,
    user_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);


-- CREATED
CREATE TABLE GENRE ( 
    genre_id INT PRIMARY KEY, 
    genre_name varchar(80)
);

-- CREATED
CREATE TABLE PERSON ( 
    person_id INT PRIMARY KEY,
    person_name varchar(120) NOT NULL
);

-- CREATED 
CREATE TABLE ROLES ( 
    role_id SERIAL PRIMARY KEY,
    role_name VARCHAR(40) UNIQUE NOT NULL
);


-- CREATED 
CREATE TABLE CONTENT_RATING ( 
    content_rating_id INT PRIMARY KEY, 
    content_rating varchar(40),
    content_rating_description TEXT
);


-- CREATED 
CREATE TABLE MOVIE ( 
    movie_id UUID PRIMARY KEY,
    movie_title varchar(500) NOT NULL,
    content_rating_id INT references CONTENT_RATING(content_rating_id),
    plot TEXT ,
    year INT,
    released DATE,
    runtime_mins INT,
    poster_link TEXT

);

CREATE TABLE movie_genre ( 
    movie_id UUID references MOVIE(movie_id),
    genre_id INT references GENRE(genre_id),
    PRIMARY KEY (movie_id, genre_id)
);

CREATE TABLE movie_person_role( 
    movie_id UUID references MOVIE(movie_id),
    person_id INT references PERSON(person_id),
    role_id INT references roles(role_id),
    PRIMARY KEY (person_id, movie_id, role_id)
);


CREATE TABLE user_rates_movie ( 
    user_id UUID references WEBSITE_USER(user_id),
    movie_id UUID NOT NULL references MOVIE(movie_id),
    rating INT NOT NULL CHECK (rating >= 0 AND rating <= 5), 
    PRIMARY KEY (user_id, movie_id)
);