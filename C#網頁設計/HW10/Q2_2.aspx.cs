using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class Q2_2 : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        HttpCookie ck = Request.Cookies["Q2"];
        if (ck == null)
        {
            lblResult.Text = "Cookie 遺失，請重新輸入。";
            return;
        }

        string name = ck["Name"];
        int principal = int.Parse(ck["Principal"]);
        double rate = double.Parse(ck["Rate"]);
        int years = int.Parse(ck["Years"]);

        double amount = principal;
        for (int i = 0; i < years; i++)
            amount *= (1 + rate);

        int total = (int)Math.Floor(amount); // 無條件捨去

        lblResult.Text =
            $"{name}您好，你儲存的本金 {principal} 元，{years} 年後可領回 {total} 元。";
    }


}