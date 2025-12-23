import axios from "axios"

const BASE_URL = "http://127.0.0.1:8000"

async function getSearchResults(search_query) { 
    var backend = `${BASE_URL}/search/${search_query}`
    try  {
        const response = await axios.get(backend);
        return response["data"];
    } catch (error) { 
        console.error(error);
        throw error;
    }
}

getSearchResults("john wick")
  .then(a => {
    console.log(a);
  })
  .catch(err => {
    console.error(err);
  });
