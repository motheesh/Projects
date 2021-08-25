function getPrediction() {
  var call = true;
  var req = {};
  var arr = $.map($("#mydiv [id]"), function (n, i) {
    var f_id = n.id;
    var f_val = $("#" + f_id).val();
    if (f_val == "") {
      call = false;
    }
    req[f_id] = f_val;
  });
  if (call) {
    $.ajax({
      type: "POST",
      url: "/predict",
      data: JSON.stringify(req),
      success: function (data) {
        $("#result").html(data["result"]);
        $("#responseTime").html(data["responseTime"]);
      },
      error: function (xhr, status, error) {
        alert("something went wrong, please try again..");
      },
      contentType: "application/json",
      dataType: "json",
    });
  } else {
    alert("Please enter all values");
  }
}
