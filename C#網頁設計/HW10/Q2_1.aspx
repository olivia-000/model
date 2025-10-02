<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Q2_1.aspx.cs" Inherits="Q2_1" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
        <div>
        姓名：<asp:TextBox ID="txtName" runat="server" /><br />
        本金：<asp:TextBox ID="txtPrincipal" runat="server" /><br />
        利率(0.01–0.20)：<asp:TextBox ID="txtRate" runat="server" /><br />
        期數(年)：
        <asp:DropDownList ID="ddlYears" runat="server"></asp:DropDownList><br />

        <asp:Button ID="btnCalc" runat="server" Text="計算本利和"
            OnClick="btnCalc_Click" /><br /><br />

        <!-- 驗證 -->
        <asp:RequiredFieldValidator ControlToValidate="txtName"
            ErrorMessage="*姓名必填" runat="server" />
        <asp:RequiredFieldValidator ControlToValidate="txtPrincipal"
            ErrorMessage="*本金必填" runat="server" />
        <asp:RequiredFieldValidator ControlToValidate="txtRate"
            ErrorMessage="*利率必填" runat="server" />
        <asp:RequiredFieldValidator ControlToValidate="ddlYears"
            InitialValue="" ErrorMessage="*期數必選" runat="server" /><br />

        <asp:RangeValidator ControlToValidate="txtPrincipal"
            MinimumValue="100" MaximumValue="30000" Type="Integer"
            ErrorMessage="*本金需介於 100–30000" runat="server" />
        <asp:RangeValidator ControlToValidate="txtRate"
            MinimumValue="0.01" MaximumValue="0.20" Type="Double"
            ErrorMessage="*利率需介於 0.01–0.20" runat="server" />
        <asp:ValidationSummary runat="server" />
    </div>
    </form>
</body>
</html>
