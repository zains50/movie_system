import "./MoviesTab.css";
import { useState } from "react"
import MoviePlate from "./../MoviePlate/MoviePlate"

export default function MovieTab({}){

    let default_arrays = []
    for (let i =0; i < 8*10; i++){
        default_arrays = [...default_arrays, i]
    }

    const [movieList, setMovieList] = useState(default_arrays)
    // const movieList = [1,2,3,4,5,6,7,8,3,123,8001]


    return (
        <div className="MoviesTab">   
        {movieList.map((movie) => (
            <MoviePlate movie_id={movie}></MoviePlate>
        ))}
        </div>
    )
}