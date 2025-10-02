<><script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script></>

$(function(){

  /* === ① Carousel 觸控滑動 === */
  if ($.fn.swipe) {                       /* 確保有載 touchSwipe 外掛才執行 */
    $("#myCarousel").swiperight(function(){ $(this).carousel('prev'); });
    $("#myCarousel").swipeleft(function(){ $(this).carousel('next'); });
  }

  /* === ② 側邊選單開關 === */
  $('#menu-toggle').on('click', function(){
    $(this).toggleClass('open');
    $('#side-nav').toggleClass('open');
    $('#menu-backdrop').toggleClass('show');
  });

  $('#menu-backdrop, #side-nav a').on('click', function(){
    $('#menu-toggle').removeClass('open');
    $('#side-nav').removeClass('open');
    $('#menu-backdrop').removeClass('show');
  });

});
