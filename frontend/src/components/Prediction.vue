<template>
    <div class="col-sm-12">
        <b-jumbotron> 
            <h1>Монгол GPT-2</h1>
                <b-container class="bv-example-row">

                    <b-row>
                        <b-col></b-col>
                        <b-col><b-form-input v-model="text" width="100px" placeholder="Угтвар" @keyup.enter="predict" style="text-align:center"></b-form-input></b-col>
                        <b-col></b-col>
                    </b-row>
                    <br>
                    <b-row>
                        <b-col></b-col>
                        <b-button @click="predict" variant="primary" href="#" onclick="alert('Түр хүлээнэ үү')">Үүсгэх</b-button>
                        <b-col></b-col>
                    </b-row>
                    </b-container>    
            <br>
            <div><b-form-textarea
      id="textarea"
      v-model="APIResult"
      disabled 
      placeholder=""
      rows="2"
      max-rows="4"
      style="font-family:serif; text-align:center"
    ></b-form-textarea>

    
</div>
        </b-jumbotron>
    </div>
</template> 

<script>
import axios from 'axios';

export default {
    name : "Prediction",
    data() {
        return {text:"",
        APIResult : ""
        };
        
    },
    methods : {
        predict() {
             axios.post("/predict", {text:this.text}, {baseURL:"http://0.0.0.0:8080/"},{headers : {"Access-Control-Allow-Origin":"*", "Content-Type" : "application/text"}})//{params: {text : this.text}}, {headers : {"Access-Control-Allow-Origin":"*", "Content-Type" : "application/text"}})
            .then(response => {this.APIResult=response.data.prediction;})
        }
    } 
}
</script>