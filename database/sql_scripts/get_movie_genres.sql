SELECT movie_genre.genre_id , GENRE.genre_name 
FROM movie_genre 
LEFT JOIN GENRE ON movie_genre.genre_id = GENRE.genre_id
WHERE movie_id=(SELECT movie_id FROM MOVIE where movie_title='John Wick');
