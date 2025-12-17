import data from "./../assets/movie_info.json" with { type: "json" };
import MoviePage from "./../MoviePage/MoviePage"
import "./MoviePlate.css"

export default function MoviePlateDemo({movie_id}){
    var movie_title = data[movie_id]["Title"]
    const k = 20
    if (movie_title.length > k){
        movie_title = movie_title.slice(0, k)
        movie_title = movie_title + "..."
    }

    const movie_poster = `/posters/${movie_id}.jpg`;
    
    const link_route = `movie/${movie_id}`

    return ( 
        <div className="MoviePlate">
            <img src={movie_poster} alt={movie_title} />
            <li><a href={link_route}>{movie_title}
            </a></li>
        </div>
    )
}





